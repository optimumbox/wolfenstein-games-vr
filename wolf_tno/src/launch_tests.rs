//! What the launcher makes of the games' manifests: their options file (`monaka_launch::games::plan`).

use crate::manifest::MANIFESTS;
use monaka_launch::games::{Values, defaults_are_options, options_file, plan_with, values_of};

/// `values` with developer mode on, which the experimental choices need.
fn in_dev_mode(mut values: Values) -> Values {
    values.set(monaka_contract::manifest::DEV_MODE.id, "1");
    values
}

#[test]
fn defaults_alone_make_a_plan() {
    defaults_are_options(MANIFESTS);
}

#[test]
fn defaults_write_the_manifests_values() {
    let tob = MANIFESTS.iter().find(|m| m.id == "wolf_tob").unwrap();
    assert_eq!(
        options_file(tob, &Values::default()),
        "disable_view_bobbing=1\nframe_rate=60\nstereo=same\ndominant_eye=right\nhead_position=lean\neye_separation=0.064\naim=hand_rig\nmove_heading=head\nfinger_tracking=1\nfinger_binary=0\ncontroller_pad=1\nhud_scale=0.35\nhud_distance=2\nworld_hud=1\nauto_pickup=1\nother_builds=0"
    );
}

#[test]
fn frame_generation_comes_with_fsr_upscaling_and_its_dlls() {
    let tno = MANIFESTS.iter().find(|m| m.id == "wolf_tno").unwrap();
    let lines = |args: &[&str], found: bool| -> Vec<String> {
        let plan = plan_with(&in_dev_mode(values_of(tno, args)), tno, None, |_| found.then(|| std::path::PathBuf::from(r"C:\DL2")));
        plan.files.into_iter().next().unwrap().1.lines().map(str::to_owned).collect()
    };
    let with = lines(&["stereo=afw", "fsr=1", "framegen=fsr"], true);
    assert!(with.iter().any(|l| l == "framegen=fsr") && with.iter().any(|l| l == r"fsr_dlls=C:\DL2"), "{with:?}");
    let no_fsr = lines(&["stereo=afw", "fsr=0", "framegen=fsr"], true);
    assert!(!no_fsr.iter().any(|l| l.starts_with("framegen=")), "offered with FSR only: {no_fsr:?}");
    let no_dlls = lines(&["stereo=afw", "fsr=1", "framegen=fsr"], false);
    assert!(no_dlls.iter().any(|l| l == "fsr=0") && !no_dlls.iter().any(|l| l.starts_with("framegen=")), "without the DLLs both go: {no_dlls:?}");
}
