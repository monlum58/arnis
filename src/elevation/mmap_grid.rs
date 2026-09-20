//! A rows x cols grid of `f32`/`f64`, backed by a memory-mapped anonymous
//! temp file instead of a resident `Vec<Vec<T>>`.
//!
//! The point: a plain `Vec<Vec<T>>`'s heap pages are anonymous memory the OS
//! can never reclaim except by swapping, so at metro scale (multiple full
//! grids alive at once across the elevation pipeline — the raw fetch grid,
//! `repair_terrain_anomalies`'s own snapshot copy, the land-cover Gaussian
//! blur buffer — each easily 1GB+ on a big bbox) they sit fully resident for
//! the whole run, stacking up. A file-backed mmap's clean pages are ordinary
//! page cache: the kernel can evict cold regions under memory pressure and
//! re-fault them back in from the backing temp file, bounding RSS the way a
//! `Vec` never can, with identical `[row][col]` read/write ergonomics via
//! `Index`/`IndexMut`, and `.par_iter()`/`.par_iter_mut()` for the same
//! row-parallel rayon usage the codebase already relies on.
use memmap2::MmapMut;
use rayon::prelude::*;
use std::fs::File;
use std::io;
use std::marker::PhantomData;
use std::mem::size_of;
use std::ops::{Index, IndexMut};

/// The plain-old-data numeric types `MmapGrid` supports. Sealed to `f32`/
/// `f64` — the two real element types the elevation pipeline uses — rather
/// than a general-purpose `bytemuck`-style abstraction nothing else needs.
pub trait GridElement: Copy + Default + Send + Sync + 'static {}
impl GridElement for f32 {}
impl GridElement for f64 {}

/// Directory the backing temp files are created in. Deliberately **not**
/// `tempfile::tempfile()`'s default (`std::env::temp_dir()`, i.e. `/tmp` on
/// Linux): on many systems, including this one, `/tmp` is itself a `tmpfs` —
/// a RAM-backed filesystem — which would make a "file-backed" mmap over it
/// backed by RAM all along, with nothing for the kernel to evict it to under
/// pressure. `dirs::cache_dir()` (`~/.cache` on Linux, matching
/// `elevation::cache::get_cache_dir`'s own choice) is real disk on ordinary
/// systems, which is the entire point of this type.
fn backing_dir() -> io::Result<std::path::PathBuf> {
    let dir = dirs::cache_dir()
        .map(|d| d.join("arnis").join("mmap-grids"))
        .unwrap_or_else(|| std::path::PathBuf::from("./arnis-mmap-grids"));
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

pub struct MmapGrid<T: GridElement> {
    mmap: MmapMut,
    // Kept alive alongside `mmap` for clarity; `tempfile::Builder::tempfile_in`
    // already unlinks the directory entry on Unix / marks delete-on-close on
    // Windows, so no manual cleanup is needed on drop.
    _file: File,
    rows: usize,
    cols: usize,
    _marker: PhantomData<T>,
}

impl<T: GridElement> MmapGrid<T> {
    /// A zero-filled `rows x cols` grid (a fresh temp file's bytes are already
    /// zero, matching `vec![vec![T::default(); cols]; rows]`'s initial state
    /// for `f32`/`f64`, whose all-zero bit pattern is `0.0`).
    pub fn new(rows: usize, cols: usize) -> io::Result<Self> {
        let cell_count = rows
            .checked_mul(cols)
            .unwrap_or_else(|| panic!("MmapGrid dimensions overflow: {rows} x {cols}"));
        let elem_size = size_of::<T>();
        // A 0-cell grid still needs a valid (non-empty) mapping; memmap2 refuses
        // to map a zero-length file. One byte is enough since no index into
        // rows=0 or cols=0 space is ever valid.
        let byte_len = (cell_count * elem_size).max(1) as u64;
        let file = tempfile::Builder::new().tempfile_in(backing_dir()?)?.into_file();
        file.set_len(byte_len)?;
        // SAFETY: `file` is a fresh temp file created and exclusively held by
        // this process; nothing else can be concurrently mutating it out from
        // under the mapping (the usual mmap-of-a-shared-file hazard).
        let mmap = unsafe { MmapMut::map_mut(&file)? };
        Ok(Self {
            mmap,
            _file: file,
            rows,
            cols,
            _marker: PhantomData,
        })
    }

    /// Builds a grid from row data, consuming it. Rows narrower than the
    /// widest row are zero-padded, matching what a freshly allocated
    /// `vec![vec![T::default(); cols]; rows]` would already read as before
    /// being filled in.
    pub fn from_rows(rows_data: Vec<Vec<T>>) -> io::Result<Self> {
        let rows = rows_data.len();
        let cols = rows_data.iter().map(Vec::len).max().unwrap_or(0);
        let mut grid = Self::new(rows, cols)?;
        for (y, row) in rows_data.into_iter().enumerate() {
            grid[y][..row.len()].copy_from_slice(&row);
        }
        Ok(grid)
    }

    pub fn len(&self) -> usize {
        self.rows
    }

    pub fn is_empty(&self) -> bool {
        self.rows == 0
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn first(&self) -> Option<&[T]> {
        if self.rows == 0 {
            None
        } else {
            Some(&self[0])
        }
    }

    pub fn iter(&self) -> MmapGridRows<'_, T> {
        MmapGridRows { grid: self, row: 0 }
    }

    /// Transposed copy (`cols x rows`), via a blocked tile sweep rather than
    /// a naive "read one element from every row per output row" walk.
    ///
    /// The naive version strides through the *entire* backing file once per
    /// output row — for a disk-backed grid that means rereading gigabytes of
    /// data over and over (this is exactly what made the vertical half of
    /// the Gaussian blur crawl for hours under real memory pressure: reading
    /// a "column" of a row-major matrix touches one value per row, spread
    /// across the whole file). Blocking bounds the working set: each
    /// `BLOCK x BLOCK` tile touches only `BLOCK` rows and `BLOCK` columns at
    /// once (a few MB), so it stays hot in ordinary page cache regardless of
    /// how large the full grid is, instead of touching the whole file.
    pub fn transpose(&self) -> Self {
        const BLOCK: usize = 512;
        let mut out = Self::new(self.cols, self.rows).expect("transpose: mmap alloc");
        let mut row_block = 0;
        while row_block < self.rows {
            let row_end = (row_block + BLOCK).min(self.rows);
            let mut col_block = 0;
            while col_block < self.cols {
                let col_end = (col_block + BLOCK).min(self.cols);
                for y in row_block..row_end {
                    let src_row = &self[y];
                    for x in col_block..col_end {
                        out[x][y] = src_row[x];
                    }
                }
                col_block = col_end;
            }
            row_block = row_end;
        }
        out
    }

    /// The whole grid as one contiguous slice, row-major — the same layout
    /// `as_flat_slice`/`as_flat_mut_slice` and `par_iter`/`par_iter_mut`
    /// below all rely on.
    pub fn as_flat_slice(&self) -> &[T] {
        // SAFETY: the full mmap is exactly `rows * cols * size_of::<T>()`
        // bytes (by construction in `new`), page-aligned, so this covers the
        // whole mapping as `rows * cols` many `T`s.
        unsafe { std::slice::from_raw_parts(self.mmap.as_ptr().cast::<T>(), self.rows * self.cols) }
    }

    pub fn as_flat_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: see `as_flat_slice`; exclusive access follows from `&mut self`.
        unsafe {
            std::slice::from_raw_parts_mut(self.mmap.as_mut_ptr().cast::<T>(), self.rows * self.cols)
        }
    }

    /// Row-parallel iteration, same shape as `Vec<Vec<T>>::par_iter()` from
    /// `rayon::prelude`. Built on `par_chunks`, so it reuses rayon's own
    /// tested splitting rather than a hand-rolled `ParallelIterator` impl.
    pub fn par_iter(&self) -> impl IndexedParallelIterator<Item = &[T]> {
        self.as_flat_slice().par_chunks(self.cols.max(1))
    }

    pub fn par_iter_mut(&mut self) -> impl IndexedParallelIterator<Item = &mut [T]> {
        let cols = self.cols.max(1);
        self.as_flat_mut_slice().par_chunks_mut(cols)
    }

    fn row_byte_range(&self, row: usize) -> std::ops::Range<usize> {
        assert!(
            row < self.rows,
            "MmapGrid row {row} out of bounds ({} rows)",
            self.rows
        );
        let elem_size = size_of::<T>();
        let start = row * self.cols * elem_size;
        start..start + self.cols * elem_size
    }
}

pub struct MmapGridRows<'a, T: GridElement> {
    grid: &'a MmapGrid<T>,
    row: usize,
}

impl<'a, T: GridElement> Iterator for MmapGridRows<'a, T> {
    type Item = &'a [T];
    fn next(&mut self) -> Option<Self::Item> {
        if self.row >= self.grid.rows {
            return None;
        }
        let row = &self.grid[self.row];
        self.row += 1;
        Some(row)
    }
}

impl<'a, T: GridElement> IntoIterator for &'a MmapGrid<T> {
    type Item = &'a [T];
    type IntoIter = MmapGridRows<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<T: GridElement> Index<usize> for MmapGrid<T> {
    type Output = [T];
    fn index(&self, row: usize) -> &[T] {
        let range = self.row_byte_range(row);
        let bytes = &self.mmap[range];
        // SAFETY: `bytes` is exactly `self.cols * size_of::<T>()` bytes taken
        // from a page-aligned mmap at a multiple-of-`size_of::<T>()` offset,
        // so it is validly sized and aligned to reinterpret as `self.cols`
        // many `T`s.
        unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<T>(), self.cols) }
    }
}

impl<T: GridElement> IndexMut<usize> for MmapGrid<T> {
    fn index_mut(&mut self, row: usize) -> &mut [T] {
        let range = self.row_byte_range(row);
        let bytes = &mut self.mmap[range];
        // SAFETY: see `Index::index`; exclusive access follows from `&mut self`.
        unsafe { std::slice::from_raw_parts_mut(bytes.as_mut_ptr().cast::<T>(), self.cols) }
    }
}

impl<T: GridElement> Clone for MmapGrid<T> {
    /// A real deep copy: a new temp file + mapping, bytes copied over. Used
    /// by `repair_terrain_anomalies`'s snapshot-per-pass pattern and by
    /// tests — kept correct rather than cheap, since a shared-mapping
    /// "clone" would silently alias mutations between copies.
    fn clone(&self) -> Self {
        let mut copy = Self::new(self.rows, self.cols).expect("MmapGrid clone: mmap alloc");
        copy.mmap.copy_from_slice(&self.mmap);
        copy
    }

    /// Refreshes an existing grid's contents from `source` in place — no new
    /// tempfile/mapping — for the "reuse one snapshot buffer across passes"
    /// pattern in `repair_terrain_anomalies`. Panics on a dimension mismatch;
    /// callers always clone-from a same-shape grid.
    fn clone_from(&mut self, source: &Self) {
        assert_eq!(
            (self.rows, self.cols),
            (source.rows, source.cols),
            "MmapGrid::clone_from: dimension mismatch"
        );
        self.mmap.copy_from_slice(&source.mmap);
    }
}

impl<T: GridElement + PartialEq> PartialEq for MmapGrid<T> {
    fn eq(&self, other: &Self) -> bool {
        self.rows == other.rows && self.cols == other.cols && self.as_flat_slice() == other.as_flat_slice()
    }
}

impl<T: GridElement + std::fmt::Debug> std::fmt::Debug for MmapGrid<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MmapGrid")
            .field("rows", &self.rows)
            .field("cols", &self.cols)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_grid_is_zero_filled_and_indexable() {
        let grid = MmapGrid::<f32>::new(3, 4).unwrap();
        assert_eq!(grid.len(), 3);
        assert_eq!(grid.cols(), 4);
        for row in grid.iter() {
            assert_eq!(row, &[0.0f32; 4]);
        }
    }

    #[test]
    fn write_then_read_round_trips_exactly() {
        let mut grid = MmapGrid::<f32>::new(5, 5).unwrap();
        for y in 0..5 {
            for x in 0..5 {
                grid[y][x] = (y * 5 + x) as f32 * 1.5;
            }
        }
        for y in 0..5 {
            for x in 0..5 {
                assert_eq!(grid[y][x], (y * 5 + x) as f32 * 1.5);
            }
        }
    }

    #[test]
    fn from_rows_matches_source_data_and_zero_pads_short_rows() {
        let grid = MmapGrid::from_rows(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0]]).unwrap();
        assert_eq!(grid.len(), 2);
        assert_eq!(grid.cols(), 3);
        assert_eq!(&grid[0], &[1.0, 2.0, 3.0]);
        assert_eq!(&grid[1], &[4.0, 5.0, 0.0]);
    }

    #[test]
    fn preserves_nan_and_infinity_sentinels() {
        let mut grid = MmapGrid::<f32>::new(1, 3).unwrap();
        grid[0][0] = f32::NAN;
        grid[0][1] = f32::INFINITY;
        grid[0][2] = f32::NEG_INFINITY;
        assert!(grid[0][0].is_nan());
        assert_eq!(grid[0][1], f32::INFINITY);
        assert_eq!(grid[0][2], f32::NEG_INFINITY);
    }

    #[test]
    fn f64_grid_round_trips_exactly() {
        let mut grid = MmapGrid::<f64>::new(4, 4).unwrap();
        for y in 0..4 {
            for x in 0..4 {
                grid[y][x] = (y * 4 + x) as f64 * std::f64::consts::PI;
            }
        }
        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(grid[y][x], (y * 4 + x) as f64 * std::f64::consts::PI);
            }
        }
    }

    #[test]
    fn par_iter_visits_every_row_exactly_once_in_order() {
        let mut grid = MmapGrid::<f64>::new(50, 3).unwrap();
        for y in 0..50 {
            grid[y].copy_from_slice(&[y as f64, y as f64 + 1.0, y as f64 + 2.0]);
        }
        let collected: Vec<Vec<f64>> = grid.par_iter().map(|row| row.to_vec()).collect();
        assert_eq!(collected.len(), 50);
        for (y, row) in collected.iter().enumerate() {
            assert_eq!(row, &[y as f64, y as f64 + 1.0, y as f64 + 2.0]);
        }
    }

    #[test]
    fn par_iter_mut_writes_are_visible_sequentially_after() {
        let mut grid = MmapGrid::<f64>::new(200, 4).unwrap();
        grid.par_iter_mut().enumerate().for_each(|(y, row)| {
            for (x, v) in row.iter_mut().enumerate() {
                *v = (y * 4 + x) as f64;
            }
        });
        for y in 0..200 {
            for x in 0..4 {
                assert_eq!(grid[y][x], (y * 4 + x) as f64);
            }
        }
    }

    #[test]
    fn clone_from_refreshes_in_place_without_reallocating() {
        let mut a = MmapGrid::<f64>::new(3, 3).unwrap();
        let mut b = MmapGrid::<f64>::new(3, 3).unwrap();
        a[1][1] = 42.0;
        b.clone_from(&a);
        assert_eq!(b[1][1], 42.0);
        a[1][1] = 7.0;
        // `b` must not alias `a`'s storage — a real copy, not a shared mapping.
        assert_eq!(b[1][1], 42.0);
    }

    #[test]
    fn is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<MmapGrid<f32>>();
        assert_send_sync::<MmapGrid<f64>>();
    }

    #[test]
    fn transpose_matches_naive_transpose_for_a_rectangular_grid() {
        let rows = 7;
        let cols = 5;
        let mut grid = MmapGrid::<f64>::new(rows, cols).unwrap();
        for y in 0..rows {
            for x in 0..cols {
                grid[y][x] = (y * cols + x) as f64;
            }
        }
        let t = grid.transpose();
        assert_eq!(t.len(), cols);
        assert_eq!(t.cols(), rows);
        for y in 0..rows {
            for x in 0..cols {
                assert_eq!(t[x][y], grid[y][x], "mismatch at ({x},{y})");
            }
        }
    }

    #[test]
    fn transpose_is_its_own_inverse() {
        let mut grid = MmapGrid::<f64>::new(1300, 900).unwrap();
        for y in 0..grid.len() {
            for x in 0..grid.cols() {
                grid[y][x] = (y as f64 * 31.0 + x as f64 * 7.0).sin();
            }
        }
        let round_tripped = grid.transpose().transpose();
        assert_eq!(round_tripped, grid);
    }
}
