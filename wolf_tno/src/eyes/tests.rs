use super::*;
use crate::manifest::MANIFESTS;

/// The run's settings as the components read them: the shared ones, the eyes', and the frame
/// rate, movement heading, view bob, arms, auto pickup and mech lever switches their components read.
fn read_all(options: &Options, manifest: &Manifest) {
    Shared::from_options(options, manifest);
    EyeSettings::from_options(options, manifest);
    crate::cooking::frame_rate::ceiling(options);
    crate::cooking::body_forward::share_out(options);
        crate::providers::hud_pieces::hides_crosshair(options);
    crate::providers::settings::no_view_bob(options, manifest);
    crate::providers::held_item::arms(options);
    crate::cooking::auto_pickup::SWITCH.on(options, manifest);
    crate::extras::mech_levers::SWITCH.on(options, manifest);
}

#[test]
fn reads_every_manifest_option() {
    for manifest in MANIFESTS {
        let unread = monaka_core::options::unread_options(manifest, |options| read_all(options, manifest));
        assert!(unread.is_empty(), "{}: not read: {unread:?}", manifest.id);
    }
}

#[test]
fn defaults_and_separation() {
    let eyes = |text: &str| EyeSettings::from_options(&Options::parse(text), &MANIFESTS[0]);
    let shared = |text: &str| Shared::from_options(&Options::parse(text), &MANIFESTS[0]);
    let config = eyes("");
    assert!(config.stereo && config.sight_lock);
    assert_eq!(config.dominant_eye, DominantEye::Right, "Monaka's default");
    assert_eq!(eyes("dominant_eye=Left").dominant_eye, DominantEye::Left);
    assert_eq!(eyes("dominant_eye=none").dominant_eye, DominantEye::None);
    let head = |ipd: f32| HeadPose { ipd, ..crate::providers::eye_site::tests::synthetic(0.0) };
    let plain = shared("");
    assert!((config.engine_separation(&plain, &HeadPose::default()) - 1.6).abs() < 1e-5);
    assert!((config.engine_separation(&plain, &head(0.07)) - 1.75).abs() < 1e-5);
    assert!((config.engine_separation(&plain, &head(0.0)) - 1.6).abs() < 1e-5, "an implausible IPD is ignored");
    let testing = shared("test_separation=0.3");
    assert!((config.engine_separation(&testing, &head(0.07)) - 7.5).abs() < 1e-5, "test_separation reaches the engine");
    assert!((config.engine_separation(&testing, &HeadPose::default()) - 7.5).abs() < 1e-5);
    let config = eyes("stereo=off");
    assert!(!config.stereo);
    assert_eq!(config.units_per_metre, UNITS_PER_METRE);
}

#[test]
fn frame_generation_is_fsrs_when_named() {
    let read = |text: &str| EyeSettings::from_options(&Options::parse(text), &MANIFESTS[0]).framegen;
    assert!(!read(""), "off by default");
    assert!(read("stereo=afw\nfsr=1\nframegen=fsr"));
    assert!(!read("framegen=off") && !read("framegen=1"), "a generator is named");
}
