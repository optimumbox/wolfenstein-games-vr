# Wolfenstein VR

Play Wolfenstein: The New Order and Wolfenstein: The Old Blood in VR. Features:
* Independent hand/gun controls with finger tracking
* The ammo counters sit on the guns (on your palms)
* Swing the knife to stab, and when equipped, hold RT and release to throw


This is a game pack for [Monaka VR](https://github.com/Mamfro/monaka_vr), which runs it.

> **Alpha.** Expect bugs and missing pieces. Until Monaka's API settles, things change often,
> and a pack made for an older Monaka may stop working with a newer one.
>
> **To play,** press Play in Monaka VR: it starts the game itself (or joins it if it is already
> running). Starting the game from Steam while SteamVR runs puts it on SteamVR's flat Theater
> screen instead. God rays, lens flares, dim shadows and screen-space reflections are switched off
> while VR runs (the game only gets them right in one eye).

## Games

| Game | Status | Controls | Aim | HUD |
|---|---|---|---|---|
| **Wolfenstein: The New Order** | Alpha | Motion controls | 6DOF | Dynamic HUD |
| **Wolfenstein: The Old Blood** | Experimental | Motion controls | 6DOF | Dynamic HUD |

Currently only the **Steam** versions are supported.

## Setting up

1. Install **Monaka VR** from its [Releases page](https://github.com/Mamfro/monaka_vr/releases) (see
   its [README](https://github.com/Mamfro/monaka_vr)).
2. Download `wolfenstein-games-vr-<version>.zip` from this repository's
   [Releases page](https://github.com/Mamfro/wolfenstein-games-vr/releases). Don't unzip it.
3. In the Monaka VR window, press **Add game pack...** and pick the zip. Both games appear in the
   list.

Then play as Monaka VR's README says: pick the game, start SteamVR, press **Start VR**. To update,
add the newer zip the same way: it replaces the old one.

## VR Controls

LT/RT: Fire weapon
LB: Sprint
RB: Weapon Wheel

A/B/X/Y: Same as in-game
Hold Y: Journal/Notes/etc screen

ADS has been removed, since you can actually do it with your own hands.

## What works

| | The New Order | The Old Blood |
|---|---|---|
| **Aim** | Head, controller | Head, controller |
| **Hands** | Tracked, with both guns when dual wielding | Tracked, with both guns when dual wielding |
| **Melee** | Swing the knife to stab; throw it along your swing | Little tested |
| **Finger tracking** | On a free hand | On a free hand |
| **Cutscenes** | Level horizon, free look; videos on a flat screen | Videos on a flat screen; not otherwise reviewed |
| **Mounted guns and the mech** | Your right controller aims them; the view stays where you got on and the stick turns it | Not reviewed |
| **Hotwiring** | With your hands | Not in this game |

## Known bugs

* The beginning of Chapter 14 is slightly bugged and BJ is not able to get off of the floor. Exiting and re-entering VR fixes it. I have not been able to track down a fix for this yet since, well, in order to debug the issue I need to exit VR, which fixes it.
* Grenades are not physically throwable yet, they simply are throwable with the right trigger when equipped. I need to make them work the same as knives.
* Blood textures and bullet hole textures may differ per eye in rare situations

## For developers

To build it yourself, clone [Monaka VR](https://github.com/Mamfro/monaka_vr) as `monaka_vr` and
[Monaka SDK](https://github.com/Mamfro/Monaka_SDK) as `monaka_sdk`, and this repository beside them
in a `monaka_adapters` folder, then run `cargo build --release` here:

```text
monaka_sdk\
monaka_vr\
monaka_adapters\
    wolfenstein-games-vr\
```

The DLL lands in `target\release`; **Add game pack...** takes a single DLL too. `tools\Package.ps1`
makes the release zip in `dist\`.

`tools\` also has the scripts used while building this: `Play-WolfTNO.ps1` starts a session with
the usual settings, and `Capture-WolfTNO.ps1` and the files in `tools\harness` run checks on the PC
without a headset. They run Monaka's `monaka_play` from the `monaka_vr` folder beside
`monaka_adapters` (or `MONAKA_ROOT`). To build the DLL straight into Monaka's own `target` (beside
its launcher), copy `cargo-config.example.toml` to `.cargo\config.toml`.

## Licence

Free software under the [GNU General Public License v3.0](LICENSE), like Monaka VR.
