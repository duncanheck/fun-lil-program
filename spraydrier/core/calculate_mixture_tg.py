import numpy as np

# Constants
TG_WATER = -135  # Glass transition temperature of water in °C
KELVIN_OFFSET = 273.15  # Conversion from °C to Kelvin

def fox_equation(weight_fractions, tg_values):
    """Calculate the Tg of a mixture using the Fox equation."""
    # Convert Tg values to Kelvin
    tg_kelvin = [tg + KELVIN_OFFSET for tg in tg_values]
    # Calculate 1/Tg_mix = Σ(w_i / Tg_i)
    inverse_tg_sum = sum(w / tg for w, tg in zip(weight_fractions, tg_kelvin))
    # Convert back to °C
    tg_mix_kelvin = 1 / inverse_tg_sum
    return tg_mix_kelvin - KELVIN_OFFSET

def gordon_taylor(tg1, tg2, w1, w2, k):
    """Calculate Tg using Gordon-Taylor equation for two components."""
    tk1 = tg1 + KELVIN_OFFSET
    tk2 = tg2 + KELVIN_OFFSET
    tk_mix = (w1 * tk1 + k * w2 * tk2) / (w1 + k * w2)
    return tk_mix - KELVIN_OFFSET

def kwei_equation(tg_gt, q, w1, w2):
    """Calculate Tg using Kwei equation, which modifies Gordon-Taylor."""
    return tg_gt + q * w1 * w2

def get_compound_inputs():
    """Get Tg values and ratios for up to 4 compounds from user."""
    compounds = []
    print("Enter data for up to 4 compounds (leave blank to stop):")
    for i in range(4):
        tg_input = input(f"Enter Tg (°C) for compound {i+1} (or press Enter to finish): ")
        if tg_input == "":
            break
        try:
            tg = float(tg_input)
            ratio = float(input(f"Enter weight ratio for compound {i+1} (e.g., 0.3 for 30%): "))
            if ratio < 0 or ratio > 1:
                print("Ratio must be between 0 and 1. Try again.")
                return None
            compounds.append((tg, ratio))
        except ValueError:
            print("Invalid input. Please enter numeric values.")
            return None
    return compounds

def main():
    # Get compound data
    compounds = get_compound_inputs()
    if not compounds or compounds is None:
        print("No valid compounds entered. Exiting.")
        return

    # Check if sum of ratios is approximately 1 (within 0.01)
    total_ratio = sum(ratio for _, ratio in compounds)
    if not 0.99 <= total_ratio <= 1.01:
        print("Sum of ratios must be approximately 1. Exiting.")
        return

    # Get k for Gordon-Taylor
    try:
        k = float(input("Enter k value for Gordon-Taylor equation (dry blend - water): "))
    except ValueError:
        print("Invalid k value. Exiting.")
        return

    # Get optional q for Kwei
    q_input = input("Enter q value for Kwei equation (optional, press Enter to skip): ")
    q = float(q_input) if q_input else None

    # Calculate Tg_dry using Fox if multiple, else single Tg
    dry_ratios = [ratio for _, ratio in compounds]
    dry_tgs = [tg for tg, _ in compounds]
    if len(compounds) > 1:
        tg_dry = fox_equation(dry_ratios, dry_tgs)
    else:
        tg_dry = dry_tgs[0]

    # Moisture levels to calculate
    moisture_levels = [0.02, 0.05, 0.10, 0.15]

    # Calculate Tg for each moisture level
    print("\nTg of mixtures at different moisture levels:")
    for moisture in moisture_levels:
        # Adjust for moisture
        dry_fraction = 1 - moisture

        # For Fox: all components including water
        weight_fractions = [ratio * dry_fraction for _, ratio in compounds] + [moisture]
        tg_values = dry_tgs + [TG_WATER]
        tg_fox = fox_equation(weight_fractions, tg_values)

        # For GT: dry + water
        tg_gt = gordon_taylor(tg_dry, TG_WATER, dry_fraction, moisture, k)

        # For Kwei if q provided
        if q is not None:
            tg_kwei = kwei_equation(tg_gt, q, dry_fraction, moisture)
        
        # Output
        print(f"Moisture {moisture*100:.0f}%:")
        print(f"  Fox: {tg_fox:.2f} °C")
        print(f"  Gordon-Taylor: {tg_gt:.2f} °C")
        if q is not None:
            print(f"  Kwei: {tg_kwei:.2f} °C")

if __name__ == "__main__":
    main()