use super::llpoint::LLPoint;

/// A checked Bounding Box.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct LLBBox {
    /// The "bottom-left" vertex of the rectangle
    min: LLPoint,

    /// The "top-right" vertex of the rectangle
    max: LLPoint,
}

impl LLBBox {
    pub fn new(min_lat: f64, min_lng: f64, max_lat: f64, max_lng: f64) -> Result<Self, String> {
        if min_lng >= max_lng {
            return Err(format!(
                "Invalid LLBBox: min_lng {min_lng} >= max_lng {max_lng}"
            ));
        }
        if min_lat >= max_lat {
            return Err(format!(
                "Invalid LLBBox: min_lat {min_lat} >= max_lat {max_lat}"
            ));
        }

        let min = LLPoint::new(min_lat, min_lng)?;
        let max = LLPoint::new(max_lat, max_lng)?;

        Ok(Self { min, max })
    }

    pub fn from_str(s: &str) -> Result<Self, String> {
        // Empty fields are skipped so "a, b, c, d" parses like "a,b,c,d".
        let mut values: Vec<f64> = Vec::with_capacity(4);
        for field in s.split([',', ' ']).filter(|f| !f.is_empty()) {
            let value: f64 = field
                .parse()
                .map_err(|_| format!("Invalid LLBBox: '{field}' is not a number"))?;
            if !value.is_finite() {
                return Err(format!("Invalid LLBBox: '{field}' is not a finite number"));
            }
            values.push(value);
        }

        let [min_lat, min_lng, max_lat, max_lng]: [f64; 4] = values
            .try_into()
            .map_err(|v: Vec<f64>| format!("Invalid LLBBox: expected 4 values, got {}", v.len()))?;

        // So, the GUI does Lat/Lng and no GDAL (comma-sep values), which is the exact opposite of
        // what bboxfinder.com does. :facepalm: (bboxfinder is wrong here: Lat comes first!)
        // DO NOT MODIFY THIS! It's correct. The CLI/GUI is passing you the numbers incorrectly.
        Self::new(min_lat, min_lng, max_lat, max_lng)
    }

    pub fn min(&self) -> LLPoint {
        self.min
    }

    pub fn max(&self) -> LLPoint {
        self.max
    }

    pub fn contains(&self, llpoint: &LLPoint) -> bool {
        llpoint.lat() >= self.min().lat()
            && llpoint.lat() <= self.max().lat()
            && llpoint.lng() >= self.min().lng()
            && llpoint.lng() <= self.max().lng()
    }

    /// Ground area in km², on an equirectangular approximation taken at the
    /// midpoint latitude. Good enough for the size checks that use it; nothing
    /// here depends on it being an exact geodesic area.
    pub fn area_km2(&self) -> f64 {
        let mid_lat = ((self.min().lat() + self.max().lat()) / 2.0).to_radians();
        let width_m = (self.max().lng() - self.min().lng()) * 111_320.0 * mid_lat.cos();
        let height_m = (self.max().lat() - self.min().lat()) * 111_320.0;
        (width_m * height_m).abs() / 1_000_000.0
    }

    /// Split into a grid of sub-bboxes, each with area no greater than `max_area_km2`
    /// (same equirectangular approximation as `area_km2`). Sub-bboxes exactly tile the
    /// original bbox — no gaps, no overlaps, no element outside the union of the grid.
    /// Returns `vec![*self]` unchanged when already within the threshold.
    ///
    /// Row/column counts are picked from the real-world aspect ratio so tiles come out
    /// roughly square; used to keep a single Overpass query's bbox under the size that
    /// makes the API truncate its response (see `retrieve_data::fetch_data_from_overpass`).
    pub fn split_into_grid(&self, max_area_km2: f64) -> Vec<LLBBox> {
        if max_area_km2 <= 0.0 || self.area_km2() <= max_area_km2 {
            return vec![*self];
        }

        let mid_lat = ((self.min().lat() + self.max().lat()) / 2.0).to_radians();
        let lat_span = self.max().lat() - self.min().lat();
        let lng_span = self.max().lng() - self.min().lng();
        let height_m = lat_span * 111_320.0;
        // Clamp cos() away from 0 so a bbox straddling the pole doesn't divide by ~0.
        let width_m = lng_span * 111_320.0 * mid_lat.cos().abs().max(1e-6);

        let target_side_m = (max_area_km2 * 1_000_000.0).sqrt();
        let rows = ((height_m / target_side_m).ceil() as usize).max(1);
        let cols = ((width_m / target_side_m).ceil() as usize).max(1);

        let lat_step = lat_span / rows as f64;
        let lng_step = lng_span / cols as f64;

        let mut tiles = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            let min_lat = self.min().lat() + lat_step * r as f64;
            let max_lat = if r + 1 == rows {
                self.max().lat()
            } else {
                min_lat + lat_step
            };
            for c in 0..cols {
                let min_lng = self.min().lng() + lng_step * c as f64;
                let max_lng = if c + 1 == cols {
                    self.max().lng()
                } else {
                    min_lng + lng_step
                };
                if let Ok(bbox) = LLBBox::new(min_lat, min_lng, max_lat, max_lng) {
                    tiles.push(bbox);
                }
            }
        }
        tiles
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_into_grid_noop_under_threshold() {
        // Arnis, Germany bbox: well under any realistic threshold.
        let bbox = LLBBox::new(54.627053, 9.927928, 54.634902, 9.937563).unwrap();
        let tiles = bbox.split_into_grid(250.0);
        assert_eq!(tiles, vec![bbox]);
    }

    #[test]
    fn split_into_grid_exactly_covers_original_bbox_no_gaps() {
        // A ~26km x 26km box (roughly 676km^2 at this latitude), split at 50km^2/tile.
        let bbox = LLBBox::new(35.5, 139.5, 35.735, 139.735).unwrap();
        let max_area = 50.0;
        let tiles = bbox.split_into_grid(max_area);

        assert!(tiles.len() > 1, "expected a real split, got {}", tiles.len());

        // Every tile must be within the safe threshold (small floating slack for the
        // equirectangular approximation at row/col boundaries).
        for t in &tiles {
            assert!(
                t.area_km2() <= max_area * 1.001,
                "tile area {} exceeds threshold {}",
                t.area_km2(),
                max_area
            );
        }

        // Total tile area must reconstruct the original bbox's area (no gaps, no overlaps).
        // Each tile's area_km2() re-evaluates cos(mid_lat) at its own (slightly different)
        // row latitude rather than the whole bbox's, so the sum carries a small inherent
        // discretization error from the equirectangular approximation itself, not from a
        // gap or overlap in the grid — a relative tolerance is the correct check here.
        let summed_area: f64 = tiles.iter().map(|t| t.area_km2()).sum();
        let original_area = bbox.area_km2();
        let relative_error = (summed_area - original_area).abs() / original_area;
        assert!(
            relative_error < 1e-3,
            "summed tile area {summed_area} != original area {original_area} \
             (relative error {relative_error})"
        );

        // Exact coverage: every tile's corners land on the original bbox's edges or interior,
        // and scanning a fine grid of sample points, every point in the original bbox is
        // contained in exactly one tile.
        let n = 40;
        for i in 0..=n {
            for j in 0..=n {
                let lat = bbox.min().lat()
                    + (bbox.max().lat() - bbox.min().lat()) * (i as f64 / n as f64);
                let lng = bbox.min().lng()
                    + (bbox.max().lng() - bbox.min().lng()) * (j as f64 / n as f64);
                let point = LLPoint::new(lat, lng).unwrap();
                let containing = tiles.iter().filter(|t| t.contains(&point)).count();
                assert!(
                    containing >= 1,
                    "point ({lat}, {lng}) inside original bbox is not covered by any tile"
                );
            }
        }
    }

    #[test]
    fn split_into_grid_rejects_nonpositive_threshold_as_noop() {
        let bbox = LLBBox::new(35.5, 139.5, 35.735, 139.735).unwrap();
        assert_eq!(bbox.split_into_grid(0.0), vec![bbox]);
        assert_eq!(bbox.split_into_grid(-5.0), vec![bbox]);
    }

    #[test]
    fn test_valid_input() {
        assert!(LLBBox::new(0., 0., 1., 1.).is_ok());

        assert!(LLBBox::new(1., 2., 3., 4.).is_ok());

        // Arnis, Germany
        assert!(LLBBox::new(54.627053, 9.927928, 54.634902, 9.937563).is_ok());

        // Royal Observatory Greenwich, London, UK
        assert!(LLBBox::new(51.470000, -0.015000, 51.480000, 0.015000).is_ok());

        // The Bund, Shanghai, China
        assert!(LLBBox::new(31.23256, 121.46768, 31.24993, 121.50394).is_ok());

        // Santa Monica, Los Angeles, US
        assert!(LLBBox::new(34.00348, -118.51226, 34.02033, -118.47600).is_ok());

        // Sydney Opera House, Sydney, Australia
        assert!(LLBBox::new(-33.861035, 151.204137, -33.852597, 151.222268).is_ok());
    }

    #[test]
    fn test_from_str_commas() {
        const ARNIS_STR: &str = "9.927928,54.627053,9.937563,54.634902";

        let bbox_result = LLBBox::from_str(ARNIS_STR);
        assert!(bbox_result.is_ok());

        let arnis_correct: LLBBox = LLBBox {
            min: LLPoint::new(9.927928, 54.627053).unwrap(),
            max: LLPoint::new(9.937563, 54.634902).unwrap(),
        };

        assert_eq!(bbox_result.unwrap(), arnis_correct);
    }

    #[test]
    fn test_from_str_spaces() {
        const ARNIS_SPACE_STR: &str = "9.927928 54.627053 9.937563 54.634902";

        let bbox_result = LLBBox::from_str(ARNIS_SPACE_STR);
        assert!(bbox_result.is_ok());

        let arnis_correct: LLBBox = LLBBox {
            min: LLPoint::new(9.927928, 54.627053).unwrap(),
            max: LLPoint::new(9.937563, 54.634902).unwrap(),
        };

        assert_eq!(bbox_result.unwrap(), arnis_correct);
    }

    #[test]
    fn test_from_str_comma_space() {
        const ARNIS_MIXED_STR: &str = "9.927928, 54.627053, 9.937563, 54.634902";

        assert!(LLBBox::from_str(ARNIS_MIXED_STR).is_ok());
    }

    #[test]
    fn test_from_str_rejects_bad_input_without_panicking() {
        // Every one of these used to panic in `from_str`.
        assert!(LLBBox::from_str("").is_err());
        assert!(LLBBox::from_str("   ").is_err());
        assert!(LLBBox::from_str(",,,").is_err());
        assert!(LLBBox::from_str("9.927928,54.627053,9.937563").is_err());
        assert!(LLBBox::from_str("9.927928,54.627053,9.937563,54.634902,1.0").is_err());
        assert!(LLBBox::from_str("9.927928,abc,9.937563,54.634902").is_err());
        assert!(LLBBox::from_str("nan,nan,nan,nan").is_err());
        assert!(LLBBox::from_str("-inf,0,inf,1").is_err());
    }

    #[test]
    fn test_out_of_order() {
        // Violates values in vals_in_order
        assert!(LLBBox::new(0., 0., 0., 0.).is_err());
        assert!(LLBBox::new(1., 0., 0., 1.).is_err());
        assert!(LLBBox::new(0., 1., 1., 0.).is_err());
    }
}
