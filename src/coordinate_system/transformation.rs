use super::cartesian::{XZBBox, XZPoint};
use super::geographic::{LLBBox, LLPoint};

/// Earth radius in meters (WGS84 spherical approximation), matching the
/// value used in `crate::projection::web_mercator`.
const EARTH_RADIUS: f64 = 6_371_000.0;

/// Internal mode discriminator so `transform_point` can dispatch between the
/// legacy linear interpolation and Web Mercator projection.
#[allow(dead_code)]
enum ProjectionMode {
    /// Existing linear-interpolation mode (no geographic projection).
    Local,
    /// Web Mercator projection with a local origin offset.
    WebMercator {
        origin_lat: f64,
        origin_lon: f64,
        scale: f64,
        cos_lat_ref: f64,
        z_offset: f64,
    },
}

/// Transform geographic space (within llbbox) to a local tangential cartesian space (within xzbbox)
pub struct CoordTransformer {
    len_lat: f64,
    len_lng: f64,
    scale_factor_x: f64,
    scale_factor_z: f64,
    min_lat: f64,
    min_lng: f64,
    mode: ProjectionMode,
}

impl CoordTransformer {
    pub fn scale_factor_x(&self) -> f64 {
        self.scale_factor_x
    }

    pub fn scale_factor_z(&self) -> f64 {
        self.scale_factor_z
    }

    /// Local-mode position of `llpoint`, before truncation.
    fn reference_f64(&self, llpoint: LLPoint) -> (f64, f64) {
        let rel_x = (llpoint.lng() - self.min_lng) / self.len_lng;
        let rel_z = 1.0 - (llpoint.lat() - self.min_lat) / self.len_lat;
        (rel_x * self.scale_factor_x, rel_z * self.scale_factor_z)
    }

    /// Inverse of the Local mode for a reference-frame block position. Used to
    /// pick chunk bboxes whose origins land exactly on chosen block coordinates.
    pub fn reference_to_latlng(&self, x: f64, z: f64) -> (f64, f64) {
        let lng = self.min_lng + x / self.scale_factor_x * self.len_lng;
        let lat = self.min_lat + (1.0 - z / self.scale_factor_z) * self.len_lat;
        (lat, lng)
    }

    pub fn llbbox_to_xzbbox(
        llbbox: &LLBBox,
        scale: f64,
    ) -> Result<(CoordTransformer, XZBBox), String> {
        let err_header = "Construct LLBBox to XZBBox transformation failed".to_string();

        if scale <= 0.0 {
            return Err(format!("{}: scale <= 0.0", err_header));
        }

        let (scale_factor_z, scale_factor_x) = geo_distance(llbbox.min(), llbbox.max());
        let scale_factor_z: f64 = scale_factor_z.floor() * scale;
        let scale_factor_x: f64 = scale_factor_x.floor() * scale;

        let xzbbox = XZBBox::rect_from_xz_lengths(scale_factor_x, scale_factor_z)
            .map_err(|e| format!("{}:\n{}", err_header, e))?;

        Ok((
            Self {
                len_lat: llbbox.max().lat() - llbbox.min().lat(),
                len_lng: llbbox.max().lng() - llbbox.min().lng(),
                scale_factor_x,
                scale_factor_z,
                min_lat: llbbox.min().lat(),
                min_lng: llbbox.min().lng(),
                mode: ProjectionMode::Local,
            },
            xzbbox,
        ))
    }

    /// Like `llbbox_to_xzbbox`, but coordinates are those of a run over
    /// `reference_bbox`, while the returned `XZBBox` covers only
    /// `processing_bbox`, the sub-area actually fetched and generated.
    ///
    /// Every point lands exactly where a single run over the whole reference
    /// area would put it, so separately generated chunks already sit in place
    /// and their region files can simply be combined. The box's origin is where
    /// the processing area starts in that frame; arnis already indexes terrain
    /// and land cover relative to the box's minimum, not to 0.
    ///
    /// With `reference_bbox == processing_bbox` this is `llbbox_to_xzbbox`.
    pub fn llbbox_to_xzbbox_with_reference(
        reference_bbox: &LLBBox,
        processing_bbox: &LLBBox,
        scale: f64,
    ) -> Result<(CoordTransformer, XZBBox), String> {
        let (transformer, _) = Self::llbbox_to_xzbbox(reference_bbox, scale)?;

        let nw = LLPoint::new(processing_bbox.max().lat(), processing_bbox.min().lng())?;
        let se = LLPoint::new(processing_bbox.min().lat(), processing_bbox.max().lng())?;
        let (x0, z0) = transformer.reference_f64(nw);
        let (x1, z1) = transformer.reference_f64(se);
        // Rounded, not truncated: chunk corners are chosen to sit exactly on a
        // block boundary, and float error must not push that one block off.
        let (min_x, min_z) = (x0.round() as i32, z0.round() as i32);
        let max_x = (x1 as i32).max(min_x);
        let max_z = (z1 as i32).max(min_z);
        let xzbbox = XZBBox::rect_from_min_max(min_x, min_z, max_x, max_z)
            .map_err(|e| format!("Failed to create XZBBox from reference transform: {e}"))?;

        Ok((transformer, xzbbox))
    }

    /// Create a `CoordTransformer` using a Web Mercator projection.
    ///
    /// The bounding box is computed by projecting all four corners of the
    /// `llbbox` and taking the axis-aligned envelope. The returned `XZBBox`
    /// represents the Minecraft world extents for the projected area.
    pub fn with_projection(
        llbbox: &LLBBox,
        scale: f64,
        projection: &dyn crate::projection::Projection,
    ) -> Result<(CoordTransformer, XZBBox), String> {
        if scale <= 0.0 {
            return Err("Scale must be > 0.0".to_string());
        }

        // Project all four corners to find the Minecraft bounding box.
        // NW corner
        let (x_nw, z_nw) = projection.forward(llbbox.max().lat(), llbbox.min().lng());
        // SE corner
        let (x_se, z_se) = projection.forward(llbbox.min().lat(), llbbox.max().lng());
        // NE corner
        let (x_ne, z_ne) = projection.forward(llbbox.max().lat(), llbbox.max().lng());
        // SW corner
        let (x_sw, z_sw) = projection.forward(llbbox.min().lat(), llbbox.min().lng());

        let x_min = x_nw.min(x_sw).min(x_ne).min(x_se).floor() as i32;
        let x_max = x_nw.max(x_sw).max(x_ne).max(x_se).ceil() as i32;
        let z_min = z_nw.min(z_sw).min(z_ne).min(z_se).floor() as i32;
        let z_max = z_nw.max(z_sw).max(z_ne).max(z_se).ceil() as i32;

        let xzbbox = XZBBox::rect_from_min_max(x_min, z_min, x_max, z_max)
            .map_err(|e| format!("Failed to create XZBBox from projection: {}", e))?;

        let origin_lat = (llbbox.min().lat() + llbbox.max().lat()) / 2.0;
        let origin_lon = (llbbox.min().lng() + llbbox.max().lng()) / 2.0;
        let cos_lat_ref = origin_lat.to_radians().cos();

        // z_offset chosen so that forward(origin_lat, _) gives z = 0.
        let z_offset = EARTH_RADIUS
            * (std::f64::consts::FRAC_PI_4 + origin_lat.to_radians() / 2.0)
                .tan()
                .ln()
            * scale;

        Ok((
            CoordTransformer {
                len_lat: llbbox.max().lat() - llbbox.min().lat(),
                len_lng: llbbox.max().lng() - llbbox.min().lng(),
                scale_factor_x: (x_max - x_min) as f64,
                scale_factor_z: (z_max - z_min) as f64,
                min_lat: llbbox.min().lat(),
                min_lng: llbbox.min().lng(),
                mode: ProjectionMode::WebMercator {
                    origin_lat,
                    origin_lon,
                    scale,
                    cos_lat_ref,
                    z_offset,
                },
            },
            xzbbox,
        ))
    }

    pub fn transform_point(&self, llpoint: LLPoint) -> XZPoint {
        match &self.mode {
            ProjectionMode::Local => {
                // Calculate the relative position within the bounding box
                let rel_x: f64 = (llpoint.lng() - self.min_lng) / self.len_lng;
                let rel_z: f64 = 1.0 - (llpoint.lat() - self.min_lat) / self.len_lat;

                // Apply scaling factors for each dimension and convert to Minecraft coordinates
                let x: i32 = (rel_x * self.scale_factor_x) as i32;
                let z: i32 = (rel_z * self.scale_factor_z) as i32;

                XZPoint::new(x, z)
            }
            ProjectionMode::WebMercator {
                origin_lon,
                scale,
                cos_lat_ref,
                z_offset,
                ..
            } => {
                let x =
                    EARTH_RADIUS * (llpoint.lng() - origin_lon).to_radians() * cos_lat_ref * scale;
                let z = -EARTH_RADIUS
                    * (std::f64::consts::FRAC_PI_4 + llpoint.lat().to_radians() / 2.0)
                        .tan()
                        .ln()
                    * scale
                    + z_offset;

                XZPoint::new(x as i32, z as i32)
            }
        }
    }
}

// (lat meters, lon meters)
#[inline]
pub fn geo_distance(a: LLPoint, b: LLPoint) -> (f64, f64) {
    let z: f64 = lat_distance(a.lat(), b.lat());

    // distance between two lons depends on their latitude. In this case we'll just average them
    let x: f64 = lon_distance((a.lat() + b.lat()) / 2.0, a.lng(), b.lng());

    (z, x)
}

// Haversine but optimized for a latitude delta of 0
// returns meters
fn lon_distance(lat: f64, lon1: f64, lon2: f64) -> f64 {
    const R: f64 = 6_371_000.0;
    let d_lon: f64 = (lon2 - lon1).to_radians();
    let a: f64 =
        lat.to_radians().cos() * lat.to_radians().cos() * (d_lon / 2.0).sin() * (d_lon / 2.0).sin();
    let c: f64 = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());

    R * c
}

// Haversine but optimized for a longitude delta of 0
// returns meters
fn lat_distance(lat1: f64, lat2: f64) -> f64 {
    const R: f64 = 6_371_000.0;
    let d_lat: f64 = (lat2 - lat1).to_radians();
    let a: f64 = (d_lat / 2.0).sin() * (d_lat / 2.0).sin();
    let c: f64 = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());

    R * c
}

// copied legacy code
// Function to convert latitude and longitude to Minecraft coordinates.
#[cfg(test)]
pub fn lat_lon_to_minecraft_coords(
    lat: f64,
    lon: f64,
    bbox: LLBBox, // (min_lon, min_lat, max_lon, max_lat)
    scale_factor_z: f64,
    scale_factor_x: f64,
) -> (i32, i32) {
    // Calculate the relative position within the bounding box
    let rel_x: f64 = (lon - bbox.min().lng()) / (bbox.max().lng() - bbox.min().lng());
    let rel_z: f64 = 1.0 - (lat - bbox.min().lat()) / (bbox.max().lat() - bbox.min().lat());

    // Apply scaling factors for each dimension and convert to Minecraft coordinates
    let x: i32 = (rel_x * scale_factor_x) as i32;
    let z: i32 = (rel_z * scale_factor_z) as i32;

    (x, z)
}

#[cfg(test)]
mod chunking_tests {
    use super::*;
    use crate::test_utilities::get_llbbox_arnis;

    /// A chunk bbox whose north-west corner sits exactly on reference block
    /// (x0, z0), running to (x1, z1).
    fn chunk_at(reference: &CoordTransformer, x0: f64, z0: f64, x1: f64, z1: f64) -> LLBBox {
        let (max_lat, min_lng) = reference.reference_to_latlng(x0, z0);
        let (min_lat, max_lng) = reference.reference_to_latlng(x1, z1);
        LLBBox::new(min_lat, min_lng, max_lat, max_lng).unwrap()
    }

    // A single run is the reference == processing case, and must not change at all.
    #[test]
    fn reference_equal_to_processing_is_the_plain_transform() {
        let bbox = get_llbbox_arnis();
        let (plain, plain_box) = CoordTransformer::llbbox_to_xzbbox(&bbox, 1.0).unwrap();
        let (with_ref, ref_box) =
            CoordTransformer::llbbox_to_xzbbox_with_reference(&bbox, &bbox, 1.0).unwrap();
        assert_eq!(
            (
                plain_box.min_x(),
                plain_box.min_z(),
                plain_box.max_x(),
                plain_box.max_z()
            ),
            (
                ref_box.min_x(),
                ref_box.min_z(),
                ref_box.max_x(),
                ref_box.max_z()
            )
        );
        for i in 0..=20 {
            for j in 0..=20 {
                let p = LLPoint::new(
                    bbox.min().lat() + (bbox.max().lat() - bbox.min().lat()) * i as f64 / 20.0,
                    bbox.min().lng() + (bbox.max().lng() - bbox.min().lng()) * j as f64 / 20.0,
                )
                .unwrap();
                assert_eq!(plain.transform_point(p), with_ref.transform_point(p));
            }
        }
    }

    // What chunked generation relies on: a chunk puts every point, bit for bit,
    // where a single run over the whole reference area would, inside the chunk
    // or not, and its box starts exactly on the block it was cut at. Otherwise
    // separately generated chunks would not line up in one world.
    #[test]
    fn chunk_coordinates_match_a_single_run_over_the_reference_area() {
        let bbox = get_llbbox_arnis();
        let (whole, _) = CoordTransformer::llbbox_to_xzbbox(&bbox, 1.0).unwrap();

        let chunks = [
            chunk_at(&whole, 0.0, 0.0, 256.0, 512.0),
            chunk_at(&whole, 256.0, 128.0, 600.0, 800.0),
        ];
        let expected_offsets = [(0, 0), (256, 128)];

        for (chunk, expected_offset) in chunks.iter().zip(expected_offsets) {
            let (local, local_box) =
                CoordTransformer::llbbox_to_xzbbox_with_reference(&bbox, chunk, 1.0).unwrap();
            assert_eq!(
                (local_box.min_x(), local_box.min_z()),
                expected_offset,
                "chunk box must start on the block it was cut at"
            );

            for i in 0..=40 {
                for j in 0..=40 {
                    let p = LLPoint::new(
                        bbox.min().lat() + (bbox.max().lat() - bbox.min().lat()) * i as f64 / 40.0,
                        bbox.min().lng() + (bbox.max().lng() - bbox.min().lng()) * j as f64 / 40.0,
                    )
                    .unwrap();
                    assert_eq!(
                        local.transform_point(p),
                        whole.transform_point(p),
                        "chunk at {expected_offset:?} disagrees with the single run on {p:?}"
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::test_utilities::get_llbbox_arnis;

    fn test_llxztransform_one_scale_one_factor(
        scale: f64,
        test_latfactor: f64,
        test_lngfactor: f64,
    ) {
        let llbbox = get_llbbox_arnis();
        let llpoint = LLPoint::new(
            llbbox.min().lat() + (llbbox.max().lat() - llbbox.min().lat()) * test_latfactor,
            llbbox.min().lng() + (llbbox.max().lng() - llbbox.min().lng()) * test_lngfactor,
        )
        .unwrap();
        let (transformer, xzbbox_new) = CoordTransformer::llbbox_to_xzbbox(&llbbox, scale).unwrap();

        // legacy xzbbox creation
        let (scale_factor_z, scale_factor_x) = geo_distance(llbbox.min(), llbbox.max());
        let scale_factor_z: f64 = scale_factor_z.floor() * scale;
        let scale_factor_x: f64 = scale_factor_x.floor() * scale;
        let xzbbox_old = XZBBox::rect_from_xz_lengths(scale_factor_x, scale_factor_z).unwrap();

        // legacy coord transform
        let (x, z) = lat_lon_to_minecraft_coords(
            llpoint.lat(),
            llpoint.lng(),
            llbbox,
            scale_factor_z,
            scale_factor_x,
        );
        // new coord transform
        let xzpoint = transformer.transform_point(llpoint);

        assert_eq!(x, xzpoint.x);
        assert_eq!(z, xzpoint.z);
        assert_eq!(xzbbox_new.min_x(), xzbbox_old.min_x());
        assert_eq!(xzbbox_new.max_x(), xzbbox_old.max_x());
        assert_eq!(xzbbox_new.min_z(), xzbbox_old.min_z());
        assert_eq!(xzbbox_new.max_z(), xzbbox_old.max_z());
    }

    // this ensures that transformer.transform_point == legacy lat_lon_to_minecraft_coords
    #[test]
    pub fn test_llxztransform() {
        test_llxztransform_one_scale_one_factor(1.0, 0.5, 0.5);
        test_llxztransform_one_scale_one_factor(3.0, 0.1, 0.2);
        test_llxztransform_one_scale_one_factor(10.0, -1.2, 2.0);
        test_llxztransform_one_scale_one_factor(0.4, 0.3, -0.2);
        test_llxztransform_one_scale_one_factor(0.1, 0.2, 0.7);
    }

    // this ensures that invalid inputs can be handled correctly
    #[test]
    pub fn test_invalid_construct() {
        let llbbox = get_llbbox_arnis();
        let obj = CoordTransformer::llbbox_to_xzbbox(&llbbox, 0.0);
        assert!(obj.is_err());

        let obj = CoordTransformer::llbbox_to_xzbbox(&llbbox, -1.2);
        assert!(obj.is_err());
    }

    // ----- Web Mercator projection mode tests -----

    #[test]
    fn test_with_projection_constructs_successfully() {
        let llbbox = get_llbbox_arnis();
        let proj = crate::projection::WebMercatorProjection::new(
            (llbbox.min().lat() + llbbox.max().lat()) / 2.0,
            (llbbox.min().lng() + llbbox.max().lng()) / 2.0,
            1.0,
        );
        let result = CoordTransformer::with_projection(&llbbox, 1.0, &proj);
        assert!(result.is_ok());
    }

    #[test]
    fn test_with_projection_invalid_scale() {
        let llbbox = get_llbbox_arnis();
        let proj = crate::projection::WebMercatorProjection::new(54.63, 9.93, 1.0);

        assert!(CoordTransformer::with_projection(&llbbox, 0.0, &proj).is_err());
        assert!(CoordTransformer::with_projection(&llbbox, -1.0, &proj).is_err());
    }

    #[test]
    fn test_with_projection_xzbbox_contains_projected_corners() {
        let llbbox = get_llbbox_arnis();
        let proj = crate::projection::WebMercatorProjection::new(
            (llbbox.min().lat() + llbbox.max().lat()) / 2.0,
            (llbbox.min().lng() + llbbox.max().lng()) / 2.0,
            1.0,
        );
        let (transformer, xzbbox) = CoordTransformer::with_projection(&llbbox, 1.0, &proj).unwrap();

        // All four corners should map inside the xzbbox
        let corners = [
            LLPoint::new(llbbox.min().lat(), llbbox.min().lng()).unwrap(),
            LLPoint::new(llbbox.min().lat(), llbbox.max().lng()).unwrap(),
            LLPoint::new(llbbox.max().lat(), llbbox.min().lng()).unwrap(),
            LLPoint::new(llbbox.max().lat(), llbbox.max().lng()).unwrap(),
        ];

        for corner in &corners {
            let pt = transformer.transform_point(*corner);
            assert!(
                pt.x >= xzbbox.min_x() && pt.x <= xzbbox.max_x(),
                "x={} out of xzbbox [{}, {}] for corner ({}, {})",
                pt.x,
                xzbbox.min_x(),
                xzbbox.max_x(),
                corner.lat(),
                corner.lng(),
            );
            assert!(
                pt.z >= xzbbox.min_z() && pt.z <= xzbbox.max_z(),
                "z={} out of xzbbox [{}, {}] for corner ({}, {})",
                pt.z,
                xzbbox.min_z(),
                xzbbox.max_z(),
                corner.lat(),
                corner.lng(),
            );
        }
    }

    #[test]
    fn test_with_projection_matches_standalone_projection() {
        // Verify that CoordTransformer in WebMercator mode produces the same
        // result as calling WebMercatorProjection::forward directly.
        let llbbox = get_llbbox_arnis();
        let origin_lat = (llbbox.min().lat() + llbbox.max().lat()) / 2.0;
        let origin_lon = (llbbox.min().lng() + llbbox.max().lng()) / 2.0;
        let proj = crate::projection::WebMercatorProjection::new(origin_lat, origin_lon, 1.0);
        let (transformer, _) = CoordTransformer::with_projection(&llbbox, 1.0, &proj).unwrap();

        let test_point = LLPoint::new(
            llbbox.min().lat() + (llbbox.max().lat() - llbbox.min().lat()) * 0.3,
            llbbox.min().lng() + (llbbox.max().lng() - llbbox.min().lng()) * 0.7,
        )
        .unwrap();

        let pt = transformer.transform_point(test_point);
        let (expected_x, expected_z) =
            crate::projection::Projection::forward(&proj, test_point.lat(), test_point.lng());

        // Integer truncation: the transformer casts with `as i32`
        assert_eq!(pt.x, expected_x as i32);
        assert_eq!(pt.z, expected_z as i32);
    }

    #[test]
    fn test_with_projection_east_increases_x() {
        let llbbox = get_llbbox_arnis();
        let proj = crate::projection::WebMercatorProjection::new(54.63, 9.93, 1.0);
        let (transformer, _) = CoordTransformer::with_projection(&llbbox, 1.0, &proj).unwrap();

        let west = LLPoint::new(54.63, 9.928).unwrap();
        let east = LLPoint::new(54.63, 9.937).unwrap();

        let pw = transformer.transform_point(west);
        let pe = transformer.transform_point(east);
        assert!(
            pe.x > pw.x,
            "east should have larger x: west.x={}, east.x={}",
            pw.x,
            pe.x,
        );
    }

    #[test]
    fn test_with_projection_north_decreases_z() {
        let llbbox = get_llbbox_arnis();
        let proj = crate::projection::WebMercatorProjection::new(54.63, 9.93, 1.0);
        let (transformer, _) = CoordTransformer::with_projection(&llbbox, 1.0, &proj).unwrap();

        let south = LLPoint::new(54.628, 9.93).unwrap();
        let north = LLPoint::new(54.634, 9.93).unwrap();

        let ps = transformer.transform_point(south);
        let pn = transformer.transform_point(north);
        assert!(
            pn.z < ps.z,
            "north should have smaller z: south.z={}, north.z={}",
            ps.z,
            pn.z,
        );
    }

    #[test]
    fn test_local_mode_unaffected_by_projection_addition() {
        // Double-check that the Local path is bit-identical to pre-change behavior.
        let llbbox = get_llbbox_arnis();
        let (transformer, _) = CoordTransformer::llbbox_to_xzbbox(&llbbox, 1.0).unwrap();

        let (scale_factor_z, scale_factor_x) = geo_distance(llbbox.min(), llbbox.max());
        let scale_factor_z = scale_factor_z.floor();
        let scale_factor_x = scale_factor_x.floor();

        let llpoint = LLPoint::new(
            llbbox.min().lat() + (llbbox.max().lat() - llbbox.min().lat()) * 0.5,
            llbbox.min().lng() + (llbbox.max().lng() - llbbox.min().lng()) * 0.5,
        )
        .unwrap();

        let (expected_x, expected_z) = lat_lon_to_minecraft_coords(
            llpoint.lat(),
            llpoint.lng(),
            llbbox,
            scale_factor_z,
            scale_factor_x,
        );

        let pt = transformer.transform_point(llpoint);
        assert_eq!(pt.x, expected_x);
        assert_eq!(pt.z, expected_z);
    }
}
