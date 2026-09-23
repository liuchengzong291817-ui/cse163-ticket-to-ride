using System;
using System.Reflection;
using Exosuit;
using RimWorld;
using Verse;

namespace MobileDragoonCEArmorFix
{
    /// <summary>
    /// Mobile Dragoon core that keeps all Exosuit Framework behavior and replaces
    /// only the armor-to-structure calculation with a CE-scale single-layer resolver.
    /// </summary>
    public class DragoonCoreCE : Exosuit_Core
    {
        private const float WeakZoneChance = 0.25f;
        private const float WeakZoneArmorFactor = 0.50f;
        private const float HardArmorDamageFactor = 0.50f;
        private const float SpikeTrapAPModifierBlunt = 0.65f;

        public override float GetPostArmorDamage(ref DamageInfo dinfo)
        {
            DamageDef damageDef = dinfo.Def;
            if (damageDef == null || damageDef.armorCategory == null)
            {
                return base.GetPostArmorDamage(ref dinfo);
            }

            StatDef armorStat = damageDef.armorCategory.armorRatingStat;
            bool isSharp = armorStat == StatDefOf.ArmorRating_Sharp;
            bool isBlunt = armorStat == StatDefOf.ArmorRating_Blunt;

            // CE heat/ambient values use a different scale, so retain framework behavior.
            if (!isSharp && !isBlunt)
            {
                return base.GetPostArmorDamage(ref dinfo);
            }

            float originalDamage = Math.Max(0f, dinfo.Amount);
            float penetration = Math.Max(0f, dinfo.ArmorPenetrationInt);
            float nominalArmor = Math.Max(0f, this.GetStatValue(armorStat));
            float effectiveArmor = nominalArmor;

            // Preserve the framework's intentional weak-zone gate, but resolve the
            // selected armor value with CE-scale continuous penetration.
            if (penetration * 2f > nominalArmor && Rand.Chance(WeakZoneChance))
            {
                effectiveArmor *= WeakZoneArmorFactor;
            }

            LayerResult primary = ResolveHardArmorLayer(
                originalDamage,
                penetration,
                effectiveArmor,
                isSharp);

            // Structural integrity represents both damaged armor and internal structure.
            float totalStructureDamage = primary.PassedDamage + primary.ArmorWear;

            // CE converts stopped/partially stopped sharp energy into blunt trauma.
            if (isSharp && originalDamage > 0f && primary.AbsorbedDamage > 0f)
            {
                float penetrationMultiplier;
                if (primary.FullyDeflected)
                {
                    penetrationMultiplier = 1f;
                }
                else if (penetration > 0f)
                {
                    float blockedPen = Math.Max(0f, penetration - primary.RemainingPenetration);
                    penetrationMultiplier =
                        (blockedPen * primary.AbsorbedDamage / originalDamage) / penetration;
                }
                else
                {
                    penetrationMultiplier = 0f;
                }

                penetrationMultiplier = Clamp01(penetrationMultiplier);
                float bluntPenetration = GetBluntPenetration(dinfo) * penetrationMultiplier;
                float bluntDamage = GetTransferredBluntDamage(
                    dinfo,
                    bluntPenetration,
                    originalDamage);

                if (bluntDamage > 0f)
                {
                    float bluntArmor = Math.Max(
                        0f,
                        this.GetStatValue(StatDefOf.ArmorRating_Blunt));
                    LayerResult blunt = ResolveHardArmorLayer(
                        bluntDamage,
                        bluntPenetration,
                        bluntArmor,
                        false);

                    totalStructureDamage += blunt.PassedDamage + blunt.ArmorWear;
                }
            }

            // Framework module HP is integral. Stochastic rounding preserves the mean.
            float result = Math.Max(0f, GenMath.RoundRandom(totalStructureDamage));
            dinfo.SetAmount(result);
            return result;
        }

        private static LayerResult ResolveHardArmorLayer(
            float damage,
            float penetration,
            float armor,
            bool isSharp)
        {
            if (damage <= 0f)
            {
                return LayerResult.Zero;
            }

            if (armor <= 0f)
            {
                return new LayerResult(damage, damage, 0f, penetration, false);
            }

            // CE treats a sharp hit as deflected when armor strictly exceeds AP.
            bool fullyDeflected = isSharp && armor > penetration;

            float passedDamage;
            float remainingPenetration;

            if (penetration <= 0f)
            {
                passedDamage = fullyDeflected ? 0f : damage;
                remainingPenetration = 0f;
            }
            else if (fullyDeflected)
            {
                passedDamage = 0f;
                remainingPenetration = penetration;
            }
            else
            {
                remainingPenetration = Math.Max(0f, penetration - armor);
                passedDamage = damage * Clamp01(remainingPenetration / penetration);
            }

            float absorbedDamage = Math.Max(0f, damage - passedDamage);
            float armorWear = 0f;

            // CE hard blunt armor takes no durability loss below half its rating.
            bool elasticBluntImpact =
                !isSharp && armor > 0f && penetration / armor < 0.5f;
            if (!elasticBluntImpact && penetration > 0f)
            {
                float absorbedWear = absorbedDamage
                    * Math.Min(1f, (penetration * penetration) / (armor * armor));
                float passedWear = passedDamage * Clamp01(armor / penetration);
                armorWear = (absorbedWear + passedWear) * HardArmorDamageFactor;
            }

            return new LayerResult(
                damage,
                passedDamage,
                armorWear,
                remainingPenetration,
                fullyDeflected);
        }

        private static float GetBluntPenetration(DamageInfo dinfo)
        {
            // ProjectilePropertiesCE.armorPenetrationBlunt, read reflectively to avoid
            // hard-linking this assembly to a particular CombatExtended.dll build.
            object projectile = dinfo.Weapon != null ? dinfo.Weapon.projectile : null;
            float reflected = ReadFloatMember(projectile, "armorPenetrationBlunt");
            if (reflected > 0f)
            {
                return reflected;
            }

            // CE stores the active melee verb because DamageInfo does not carry ToolCE.
            try
            {
                Type verbType = Type.GetType(
                    "CombatExtended.Verb_MeleeAttackCE, CombatExtended",
                    false);
                PropertyInfo lastAttack = verbType != null
                    ? verbType.GetProperty(
                        "LastAttackVerb",
                        BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static)
                    : null;
                object verb = lastAttack != null ? lastAttack.GetValue(null, null) : null;
                reflected = ReadFloatMember(verb, "ArmorPenetrationBlunt");
                if (reflected > 0f)
                {
                    return reflected;
                }
            }
            catch
            {
                // Continue to safe fallbacks when CE internals differ.
            }

            if (dinfo.Instigator != null
                && dinfo.Instigator.def != null
                && dinfo.Instigator.def.thingClass == typeof(Building_TrapDamager))
            {
                return Math.Max(
                    0f,
                    dinfo.Instigator.GetStatValue(StatDefOf.TrapMeleeDamage, true)
                    * SpikeTrapAPModifierBlunt);
            }

            return Math.Max(
                0f,
                dinfo.Def != null ? dinfo.Def.defaultArmorPenetration : 0f);
        }

        private static float GetTransferredBluntDamage(
            DamageInfo dinfo,
            float bluntPenetration,
            float originalDamage)
        {
            if (bluntPenetration <= 0f)
            {
                return 0f;
            }

            // Same penetration-to-blunt-damage relationship used by CE deflection.
            float bluntDamage =
                (float)Math.Pow(bluntPenetration * 10000f, 1f / 3f) / 10f;

            object projectile = dinfo.Weapon != null ? dinfo.Weapon.projectile : null;
            if (projectile != null)
            {
                float projectileBaseDamage = ReadFloatMember(
                    projectile,
                    "damageAmountBase");
                if (projectileBaseDamage > 0f)
                {
                    bluntDamage *= originalDamage / projectileBaseDamage;
                }
            }

            return Math.Max(0f, bluntDamage);
        }

        private static float ReadFloatMember(object target, string memberName)
        {
            if (target == null)
            {
                return 0f;
            }

            Type type = target.GetType();
            const BindingFlags Flags =
                BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance;

            FieldInfo field = type.GetField(memberName, Flags);
            if (field != null)
            {
                return ConvertToFloat(field.GetValue(target));
            }

            PropertyInfo property = type.GetProperty(memberName, Flags);
            if (property != null)
            {
                return ConvertToFloat(property.GetValue(target, null));
            }

            return 0f;
        }

        private static float ConvertToFloat(object value)
        {
            if (value == null)
            {
                return 0f;
            }

            try
            {
                return Convert.ToSingle(value);
            }
            catch
            {
                return 0f;
            }
        }

        private static float Clamp01(float value)
        {
            if (value <= 0f)
            {
                return 0f;
            }

            return value >= 1f ? 1f : value;
        }

        private struct LayerResult
        {
            public static readonly LayerResult Zero =
                new LayerResult(0f, 0f, 0f, 0f, false);

            public LayerResult(
                float originalDamage,
                float passedDamage,
                float armorWear,
                float remainingPenetration,
                bool fullyDeflected)
            {
                PassedDamage = Math.Max(0f, passedDamage);
                ArmorWear = Math.Max(0f, armorWear);
                RemainingPenetration = Math.Max(0f, remainingPenetration);
                FullyDeflected = fullyDeflected;
                AbsorbedDamage = Math.Max(0f, originalDamage - PassedDamage);
            }

            public float PassedDamage;
            public float ArmorWear;
            public float RemainingPenetration;
            public bool FullyDeflected;
            public float AbsorbedDamage;
        }
    }

    [StaticConstructorOnStartup]
    internal static class Bootstrap
    {
        static Bootstrap()
        {
            Log.Message(
                "[Mobile Dragoon CE Armor Fix] Loaded CE-scale core armor resolver.");
        }
    }
}
