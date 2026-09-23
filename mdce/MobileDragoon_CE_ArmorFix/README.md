# Mobile Dragoon - CE Armor Resolver Fix

RimWorld 1.6 compatibility patch for Combat Extended, Exosuit Framework, The Dead Man's Switch, and MobileDragoon.

## Installation

Extract `MobileDragoon_CE_ArmorFix` into `RimWorld/Mods/`, enable it, and place it after its required mods. Restart RimWorld. Back up the save before adding or removing combat DLL mods.

## Changes

- Only DMS Mobile Dragoon cores use the replacement resolver.
- Retains the framework's 25% weak-zone roll when penetration exceeds half of nominal armor; a weak-zone hit uses 50% effective armor.
- Replaces vanilla 0/50/100% armor rolls with CE-scale continuous penetration and hard-armor wear.
- Transfers stopped or partially stopped sharp energy into blunt damage using CE projectile/melee blunt penetration when available.
- Keeps shields, structural integrity, module HP distribution, low-integrity pilot leakage, destruction, and wreckage behavior.
- Leaves heat and unusual armor categories on the original framework calculation because those CE values do not use the mm-RHA scale.
- Changes PV-4 core sharp armor from 45 to 57 mm RHA.

## Expected result

For a 57 mm RHA PV-4 core, penetration differences now matter continuously. A 55-AP hit and a 29-AP hit no longer collapse into the same 75% block / 25% full-damage result. The original weak-zone concept remains, but both the main-armor and weak-zone branches use CE-scaled resolution.

## Uninstalling

Changing the `thingClass` of existing apparel can be unsafe when removing a mod mid-save. Back up the save and remove Mobile Dragoon cores from pawns before disabling this patch.
