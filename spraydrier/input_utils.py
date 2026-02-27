# input_utils.py

def get_optional_float(prompt):
    user_input = input(prompt).strip()
    if user_input == "":
        return "Not Available"
    try:
        value = float(user_input.replace(',', '.'))
        if value <= 0:
            raise ValueError(f"{prompt.strip()} must be positive")
        return value
    except ValueError as e:
        if "must be positive" in str(e):
            raise
        return "Not Available"


def collect_inputs():
    inputs = {}
    try:
        # Batch id should be a short identifier. Guard against the common
        # accidental paste where the terminal inserts the last command (e.g.
        # "/Users/.../python /path/to/main.py") as the batch id. Re-prompt
        # until a reasonable batch id is entered.
        while True:
            batch_id_raw = input("Enter Batch Id: ").strip()
            if batch_id_raw == "":
                print("Batch Id cannot be empty. Please enter a short identifier (e.g. 'test1').")
                continue
            # detect obvious pasted commands or file paths
            lower = batch_id_raw.lower()
            if '/' in batch_id_raw or lower.endswith('.py') or 'python' in lower or 'my_env' in lower:
                print("Detected a command or filepath pasted as Batch Id. Please enter a short batch identifier (not a path or command).")
                continue
            batch_id = batch_id_raw.strip().lower()
            inputs["batch_id"] = batch_id
            break
        dryer = input("Type of Dryer (e.g. B290, B300, ProCept, etc): ").strip().lower()
        inputs["dryer"] = dryer
        if dryer == "procept":
            V_chamber_m3 = 0.006686
        elif dryer == "other":
            V_chamber_m3 = float(input("Spray dryer chamber volume (m3): "))
        else:
            V_chamber_m3 = 0.00212
        inputs["V_chamber_m3"] = V_chamber_m3
        cyclone_type = input("Type of cyclone (e.g. high or std): ").strip().lower()
        inputs["cyclone_type"] = cyclone_type

        print("Enter Drying Gas Data:")
        gas1 = input(" Drying gas (air/nitrogen): ").strip().lower()
        inputs["gas1"] = gas1
        if gas1 not in ["air", "nitrogen", "n2"]:
            raise ValueError("Drying gas must be 'air' or 'nitrogen'")
        T1_C = float(input(" Drying gas temperature (Â°C): "))
        inputs["T1_C"] = T1_C
        if T1_C < 0 or T1_C > 150:
            raise ValueError("Drying gas temperature must be between 0 and 150Â°C")
        RH1 = float(input(" RH of drying gas before heating (%): "))
        inputs["RH1"] = RH1
        if not 0 <= RH1 <= 100:
            raise ValueError("RH must be between 0 and 100%")
        m1_m3ph = float(input(" Drying gas volumetric flow (mÂ³/h): "))
        inputs["m1_m3ph"] = m1_m3ph
        if m1_m3ph <= 0:
            raise ValueError("Drying gas flow must be positive")

        print("Enter Atomizing Gas Data:")
        gas2 = input(" Atomizing gas (air/nitrogen): ").strip().lower()
        inputs["gas2"] = gas2
        if gas2 not in ["air", "nitrogen", "n2"]:
            raise ValueError("Atomizing gas must be 'air' or 'nitrogen'")
        T2_C = float(input(" Atomizing gas temperature (Â°C): "))
        inputs["T2_C"] = T2_C
        if T2_C < -110 or T2_C > 90:
            raise ValueError("Atomizing gas temperature must be between -110 and 90Â°C")
        RH2 = float(input(" RH of atomizing gas (%): "))
        inputs["RH2"] = RH2
        if not 0 <= RH2 <= 100:
            raise ValueError("RH must be between 0 and 100%")
        atom_pressure = float(input(" Atomizing gas pressure (bar): "))
        inputs["atom_pressure"] = atom_pressure
        if atom_pressure <= 0:
            raise ValueError("Atomizing pressure must be positive")

        nozzle_tip_d_mm = float(input(" Nozzle tip (mm): "))
        inputs["nozzle_tip_d_mm"] = nozzle_tip_d_mm
        if nozzle_tip_d_mm <= 0:
            raise ValueError("Nozzle tip diameter must be positive")
        nozzle_cap_d_mm = float(input(" Nozzle cap diameter (mm): "))
        inputs["nozzle_cap_d_mm"] = nozzle_cap_d_mm
        if nozzle_cap_d_mm <= 0:
            raise ValueError("Nozzle cap diameter must be positive")
        nozzle_level = input(" Nozzle tip and cap are level (Y/N)?: ").strip().lower()
        inputs["nozzle_level"] = nozzle_level
        if nozzle_level not in ["y", "n"]:
            raise ValueError("Nozzle level must be 'Y' or 'N'")

        print("Enter Outlet Temperature:")
        T_outlet_C = float(input("Enter outlet gas temperature (Â°C): "))
        inputs["T_outlet_C"] = T_outlet_C
        if T_outlet_C < -110 or T_outlet_C > 90:
            raise ValueError("Outlet temperature must be between -110 and 90Â°C")

        # Optional: measured outlet relative humidity
        measured = input("Measured outlet RH (%, leave blank if not available): ").strip()
        if measured == "":
            inputs["measured_RH_out"] = "Not Available"
        else:
            try:
                mrh = float(measured)
                if not 0 <= mrh <= 100:
                    raise ValueError("Measured RH must be between 0 and 100%")
                inputs["measured_RH_out"] = mrh
            except ValueError:
                inputs["measured_RH_out"] = "Not Available"

        print("Enter Feed Solution Properties:")
        ds = input(" Drug Substance (e.g., IgG): ").strip().lower()
        inputs["ds"] = ds
        ds_conc = float(input(" Drug substance conc. (mg/mL): "))
        inputs["ds_conc"] = ds_conc
        if ds_conc < 0:
            raise ValueError("Drug substance concentration must be non-negative")
        ds_mw = float(input(" Drug substance mw (kDa - eg 63, 150): "))
        inputs["ds_mw"] = ds_mw
        # Optional: allow user to provide glass transition temperature (Tg) for the drug substance
        ds_tg_raw = input(" Drug substance Tg (Â°C) - leave blank to skip: ").strip()
        if ds_tg_raw == "":
            inputs["ds_Tg_C"] = None
        else:
            try:
                inputs["ds_Tg_C"] = float(ds_tg_raw.replace(',', '.'))
            except Exception:
                inputs["ds_Tg_C"] = None
        if ds_mw > 250:
            ds_mw = float(input(" Re-enter Drug substance mw (kDa - eg 63, 150): "))
            inputs["ds_mw"] = ds_mw
        if ds_mw <= 0:
            raise ValueError("Drug substance MW must be positive")

        # Offer the user an option to enter viscosity manually (in cP). If not, use existing logic.
        vis_override = input("Do you want to enter viscosity? (Y/N)?: ").strip().lower()
        if vis_override == "y":
            # user provides viscosity in cP (centipoise) -> convert to PaÂ·s (1 cP = 0.001 PaÂ·s)
            visc_cP = float(input(" Enter feed solution viscosity (cP): "))
            viscosity = visc_cP * 0.001
            if viscosity <= 0:
                raise ValueError("Viscosity must be positive")
            inputs["viscosity"] = viscosity
            inputs["viscosity_user_input"] = True
        else:
            if ds in ["pgt121", "igg"]:
                viscosity = 0.00003 * ds_conc - 0.0003
                inputs["viscosity"] = viscosity
            else:
                viscosity = float(input(" Viscosity of spray dryer feed solution without MoNi (PaÂ·s): "))
                inputs["viscosity"] = viscosity
                if viscosity <= 0:
                    raise ValueError("Viscosity must be positive")
            inputs["viscosity_user_input"] = False

        # Offer the user an option to enter surface tension manually (in mN/m). If not, use existing logic.
        st_override = input("Do you want to enter surface tension? (Y/N)?: ").strip().lower()
        if st_override == "y":
            # user provides surface tension in mN/m -> convert to N/m (1 mN/m = 0.001 N/m)
            st_mN = float(input(" Enter feed solution surface tension (mN/m): "))
            surface_tension = st_mN * 0.001
            if surface_tension <= 0:
                raise ValueError("Surface tension must be positive")
            inputs["surface_tension"] = surface_tension
            inputs["surface_tension_user_input"] = True
        else:
            if ds_mw <= 35:
                surface_tension = float(input("Surface tension of Feed solution without MoNi (N/m): "))
                inputs["surface_tension"] = surface_tension
                if surface_tension <= 0:
                    raise ValueError("Surface tension must be positive")
                inputs["surface_tension_user_input"] = True
            elif 35 < ds_mw <= 100:
                surface_tension = -4e-5 * ds_conc + 0.074
                inputs["surface_tension_user_input"] = False
            else:
                surface_tension = -6.5e-5 * ds_conc + 0.068
                inputs["surface_tension_user_input"] = False
            inputs["surface_tension"] = surface_tension

        # Diffusion coefficient is now calculated automatically in simulation.py
        inputs["D_solute"] = 1e-10  # Placeholder, will be overridden

        # Accept solids fraction as percent (e.g., enter 5 or 5% for 5%) or as a
        # fraction (e.g., 0.05). Prefer percent input for clarity.
        solids_frac_input = input(" Solids fraction in feed (percent e.g. 5 or 5% or fraction e.g. 0.05): ").strip()
        try:
            if '%' in solids_frac_input:
                solids_frac = float(solids_frac_input.rstrip('%')) / 100.0
            else:
                val = float(solids_frac_input)
                if val >= 1:
                    solids_frac = val / 100.0
                else:
                    solids_frac = val
        except ValueError:
            raise ValueError("Invalid solids fraction input. Enter as number or number%.")
        inputs["solids_frac"] = solids_frac
        if solids_frac <= 0 or solids_frac >= 1:
            raise ValueError("Solids fraction must be between 0 and 1 (entered as percent or fraction)")

        moni_conc = float(input(" Moni conc. (mg/mL): "))
        inputs["moni_conc"] = moni_conc
        if moni_conc < 0:
            raise ValueError("Moni concentration must be non-negative")

        pH_question = input("Feed solution pH known? (Y/N): ").strip().lower()
        inputs["pH_question"] = pH_question
        if pH_question == "y":
            pH = input(" Feed solution pH: ")
            inputs["pH"] = pH
        else:
            pH = "not known"
            inputs["pH"] = pH

        buffer_question = input("Feed solution buffered? (Y/N): ").strip().lower()
        inputs["buffer_question"] = buffer_question
        if buffer_question == "y":
            buffer = input(" Buffer added: ")
            inputs["buffer"] = buffer
            buffer_conc = float(input(" Buffer concentration (mg/mL): "))
            inputs["buffer_conc"] = buffer_conc
            # Robust MW prompt: allow blank to skip, otherwise require positive float
            while True:
                buf_mw_raw = input(" Buffer MW (Da) - leave blank to skip or enter numeric (e.g. 136 for acetate, 342 for trehalose): ").strip()
                if buf_mw_raw == "":
                    inputs["buffer_mw"] = None
                    break
                try:
                    buf_mw_val = float(buf_mw_raw.replace(',', '.'))
                    if buf_mw_val <= 0:
                        print("Buffer MW must be a positive number. Try again or leave blank to skip.")
                        continue
                    inputs["buffer_mw"] = buf_mw_val
                    break
                except ValueError:
                    print("Invalid number. Enter numeric MW in Da (e.g. 136) or leave blank to skip.")
            # Optional buffer Tg
            buf_tg_raw = input(" Buffer Tg (Â°C) - leave blank to skip: ").strip()
            if buf_tg_raw == "":
                inputs["buffer_Tg_C"] = None
            else:
                try:
                    inputs["buffer_Tg_C"] = float(buf_tg_raw.replace(',', '.'))
                except Exception:
                    inputs["buffer_Tg_C"] = None
        else:
            buffer = "not buffered"
            buffer_conc = 0
            inputs["buffer"] = buffer
            inputs["buffer_conc"] = buffer_conc

        materials = input(" Additional excipients in formulation? (Y/N): ").strip().lower()
        inputs["materials"] = materials
        if materials == "y":
            number = int(input(" Number of stabilizers/additives: (1-3): "))
            inputs["number_excipients"] = number
            if number not in [1, 2, 3]:
                raise ValueError("Number of stabilizers must be 1, 2, or 3")
            if number == 1:
                stabilizer_A = input(" Stabilizer name (e.g. trehalose, etc): ").strip().lower()
                stab_A_conc = float(input(" Stabilizer conc (mg/mL): "))
                # Stabilizer MW prompt - robust
                while True:
                    stab_mw_raw = input(" Stabilizer MW (Da) - leave blank to skip or enter numeric (e.g. 342 for trehalose): ").strip()
                    if stab_mw_raw == "":
                        stab_A_mw = None
                        break
                    try:
                        stab_A_mw = float(stab_mw_raw.replace(',', '.'))
                        if stab_A_mw <= 0:
                            print("Stabilizer MW must be positive. Try again or leave blank to skip.")
                            continue
                        break
                    except ValueError:
                        print("Invalid number. Enter numeric MW in Da or leave blank to skip.")
                inputs["stab_A_mw"] = stab_A_mw
                # Optional stabilizer Tg
                stab_tg_raw = input(" Stabilizer Tg (Â°C) - leave blank to skip: ").strip()
                if stab_tg_raw == "":
                    inputs["stab_A_Tg_C"] = None
                else:
                    try:
                        inputs["stab_A_Tg_C"] = float(stab_tg_raw.replace(',', '.'))
                    except Exception:
                        inputs["stab_A_Tg_C"] = None
                additive_B = 'none'
                additive_B_conc = 'none'
                additive_C = 'none'
                additive_C_conc = 'none'
            elif number == 2:
                stabilizer_A = input(" Stabilizer name (e.g. trehalose, etc): ").strip().lower()
                stab_A_conc = float(input(" Stabilizer conc (mg/mL): "))
                try:
                    stab_A_mw = float(input(" Stabilizer MW (Da, enter approximate MW e.g. 342 for trehalose): "))
                except Exception:
                    stab_A_mw = None
                inputs["stab_A_mw"] = stab_A_mw
                # Optional stabilizer Tg
                try:
                    stab_tg_raw = input(" Stabilizer Tg (Â°C) - leave blank to skip: ").strip()
                    inputs["stab_A_Tg_C"] = float(stab_tg_raw.replace(',', '.')) if stab_tg_raw != "" else None
                except Exception:
                    inputs["stab_A_Tg_C"] = None
                additive_B = input(" Additive name (e.g. L-Histidine, etc): ").strip().lower()
                # Allow blank additive name -> treat as 'none'
                if additive_B == "":
                    additive_B = 'none'
                    additive_B_conc = 'none'
                    additive_B_mw = None
                else:
                    # Use optional float parsing so blank input doesn't raise
                    try:
                        addb_con_raw = input(" Additive conc (mg/mL): ").strip()
                        if addb_con_raw == "":
                            additive_B_conc = 0.0
                        else:
                            additive_B_conc = float(addb_con_raw.replace(',', '.'))
                    except Exception:
                        additive_B_conc = 0.0
                # Additive B MW prompt - robust
                while True:
                    addb_mw_raw = input(" Additive B MW (Da) - leave blank to skip or enter numeric: ").strip()
                    if addb_mw_raw == "":
                        additive_B_mw = None
                        break
                    try:
                        additive_B_mw = float(addb_mw_raw.replace(',', '.'))
                        if additive_B_mw <= 0:
                            print("Additive B MW must be positive. Try again or leave blank to skip.")
                            continue
                        break
                    except ValueError:
                        print("Invalid number. Enter numeric MW in Da or leave blank to skip.")
                inputs["additive_B_mw"] = additive_B_mw
                additive_C = 'none'
                additive_C_conc = 'none'
            else:
                stabilizer_A = input(" Stabilizer name (e.g. trehalose, etc): ").strip().lower()
                stab_A_conc = float(input(" Stabilizer conc (mg/mL): "))
                try:
                    stab_A_mw = float(input(" Stabilizer MW (Da, enter approximate MW e.g. 342 for trehalose): "))
                except Exception:
                    stab_A_mw = None
                inputs["stab_A_mw"] = stab_A_mw
                # Optional stabilizer Tg
                try:
                    stab_tg_raw = input(" Stabilizer Tg (Â°C) - leave blank to skip: ").strip()
                    inputs["stab_A_Tg_C"] = float(stab_tg_raw.replace(',', '.')) if stab_tg_raw != "" else None
                except Exception:
                    inputs["stab_A_Tg_C"] = None
                additive_B = input(" Additive name (e.g. L-Histidine, etc): ").strip().lower()
                additive_B_conc = float(input(" Additive conc (mg/mL): "))
                try:
                    additive_B_mw = float(input(" Additive B MW (Da): "))
                except Exception:
                    additive_B_mw = None
                inputs["additive_B_mw"] = additive_B_mw
                additive_C = input(" Additive name (e.g. L-Histidine, etc): ").strip().lower()
                additive_C_conc = float(input(" Additive conc (mg/mL): "))
                # Additive C MW prompt - robust
                while True:
                    addc_mw_raw = input(" Additive C MW (Da) - leave blank to skip or enter numeric: ").strip()
                    if addc_mw_raw == "":
                        additive_C_mw = None
                        break
                    try:
                        additive_C_mw = float(addc_mw_raw.replace(',', '.'))
                        if additive_C_mw <= 0:
                            print("Additive C MW must be positive. Try again or leave blank to skip.")
                            continue
                        break
                    except ValueError:
                        print("Invalid number. Enter numeric MW in Da or leave blank to skip.")
                inputs["additive_C_mw"] = additive_C_mw
                # Optional Additive B Tg
                addb_tg_raw = input(" Additive B Tg (Â°C) - leave blank to skip: ").strip()
                if addb_tg_raw == "":
                    inputs["additive_B_Tg_C"] = None
                else:
                    try:
                        inputs["additive_B_Tg_C"] = float(addb_tg_raw.replace(',', '.'))
                    except Exception:
                        inputs["additive_B_Tg_C"] = None
                # Optional Additive C Tg
                addc_tg_raw = input(" Additive C Tg (Â°C) - leave blank to skip: ").strip()
                if addc_tg_raw == "":
                    inputs["additive_C_Tg_C"] = None
                else:
                    try:
                        inputs["additive_C_Tg_C"] = float(addc_tg_raw.replace(',', '.'))
                    except Exception:
                        inputs["additive_C_Tg_C"] = None
        else:
            stabilizer_A = 'none'
            stab_A_conc = 'none'
            additive_B = 'none'
            additive_B_conc = 'none'
            additive_C = 'none'
            additive_C_conc = 'none'
            # Ensure Tg keys exist even when no excipients provided
            inputs["stab_A_Tg_C"] = None
            inputs["additive_B_Tg_C"] = None
            inputs["additive_C_Tg_C"] = None
        inputs["stabilizer_A"] = stabilizer_A
        inputs["stab_A_conc"] = stab_A_conc
        inputs["additive_B"] = additive_B
        inputs["additive_B_conc"] = additive_B_conc
        inputs["additive_C"] = additive_C
        inputs["additive_C_conc"] = additive_C_conc

        # Option to calculate density
        calc_density_input = input("Calculate solution density from composition? (Y/N, default Y): ").strip().lower()
        inputs["calc_density"] = calc_density_input not in ('n', 'no')

        # Option to calculate diffusion coefficient
        calc_diffusion_input = input("Enter diffusion coefficient manually? (Y/N, default N): ").strip().lower()
        inputs["calc_diffusion"] = calc_diffusion_input not in ('y', 'yes')
        if not inputs["calc_diffusion"]:
            D_solute = float(input("Diffusion coefficient of solute in feed (mÂ²/s): "))
            inputs["D_solute"] = D_solute
            if D_solute <= 0:
                raise ValueError("Diffusion coefficient must be positive")
        else:
            inputs["D_solute"] = 1e-10  # Placeholder, will be calculated

        while True:
            feed = input(" Feed rate in g/min (Y/N): ").strip().lower()
            inputs["feed"] = feed
            if feed in ["y", "n"]:
                break
            print("Invalid input. Please enter 'y' or 'n'")
        if feed == "n":
            feed_mL_min = float(input(" Feed flow rate (mL/min): "))
            inputs["feed_mL_min"] = feed_mL_min
            if feed_mL_min <= 0:
                raise ValueError("Feed flow rate must be positive")
            if not inputs["calc_density"]:
                rho_l = float(input(" Feed solution density (g/mL): "))
                if rho_l <= 0:
                    raise ValueError("Feed solution density must be positive")
            else:
                rho_l = 1.0  # Placeholder, will be calculated
            inputs["rho_l"] = rho_l
        else:
            feed_g_min = float(input(" Feed flow rate (g/min): "))
            inputs["feed_g_min"] = feed_g_min
            if feed_g_min <= 0:
                raise ValueError("Feed flow rate must be positive")
            if not inputs["calc_density"]:
                rho_l = float(input("Density of solution: "))
                if rho_l <= 0:
                    raise ValueError("Solution density must be positive")
            else:
                rho_l = 1.0  # Placeholder
            inputs["rho_l"] = rho_l

        # Optional: measured powder moisture values (moved earlier)
        # Accept percent input for measured total powder moisture (e.g., 23 for 23%) and store as fraction
        total_m = input("Measured total powder moisture (percent e.g. 23, or fraction e.g. 0.23) leave blank if not available: ").strip()
        if total_m == "":
            inputs["measured_total_moisture"] = "Not Available"
        else:
            try:
                tm_val = float(total_m)
                # accept either percent (>1) or fraction (<=1)
                if tm_val > 1:
                    inputs["measured_total_moisture"] = tm_val / 100.0
                else:
                    inputs["measured_total_moisture"] = tm_val
            except ValueError:
                inputs["measured_total_moisture"] = "Not Available"

        # Accept percent input for measured powder surface moisture (e.g., 1 for 1%) and store as fraction
        surface_m = input("Measured powder surface moisture (percent e.g. 1, or fraction e.g. 0.01) leave blank if not available: ").strip()
        if surface_m == "":
            inputs["measured_surface_moisture"] = "Not Available"
        else:
            try:
                sm_val = float(surface_m)
                if sm_val > 1:
                    inputs["measured_surface_moisture"] = sm_val / 100.0
                else:
                    inputs["measured_surface_moisture"] = sm_val
            except ValueError:
                inputs["measured_surface_moisture"] = "Not Available"

        # Optional: bound moisture (fixed) percent to add to surface moisture to obtain total powder moisture
        bound_m = input("Fixed bound moisture to add to surface moisture (percent e.g. 5, or fraction e.g. 0.05). Enter 0 or leave blank to skip: ").strip()
        if bound_m == "" or bound_m == "0":
            # Not provided by user - defer potential inference below
            inputs["measured_bound_moisture"] = "Not Available"
        else:
            try:
                bm_val = float(bound_m)
                if bm_val > 1:
                    inputs["measured_bound_moisture"] = bm_val / 100.0
                else:
                    inputs["measured_bound_moisture"] = bm_val
            except ValueError:
                inputs["measured_bound_moisture"] = "Not Available"

        # If user provided total and surface but did NOT provide bound, infer bound = total - surface
        tpm = inputs.get("measured_total_moisture", "Not Available")
        spm = inputs.get("measured_surface_moisture", "Not Available")
        bmp = inputs.get("measured_bound_moisture", "Not Available")
        try:
            if bmp == "Not Available" and isinstance(tpm, (int, float)) and isinstance(spm, (int, float)):
                inferred = tpm - spm
                # clamp to non-negative and not exceed total
                inferred = max(0.0, min(inferred, tpm))
                inputs["measured_bound_moisture"] = inferred
                print(f"Info: Inferred bound moisture = {inferred*100:.3f}% (total - surface)")
        except Exception:
            # keep existing measured_bound_moisture if inference fails
            pass

        # Default to letting the model predict powder moisture unless other
        # measured values are provided earlier. This keeps behavior consistent
        # with previous default ('n' for user not providing moisture content).
        inputs["moisture_input"] = "n"
        inputs["moisture_content"] = "predicted"
        D10_actual = get_optional_float("D10 actual: ")
        inputs["D10_actual"] = D10_actual
        D50_actual = get_optional_float("D50 actual: ")
        inputs["D50_actual"] = D50_actual
        D90_actual = get_optional_float("D90 actual: ")
        inputs["D90_actual"] = D90_actual
        Span = None
        if all(isinstance(x, (int, float)) for x in [D10_actual, D50_actual, D90_actual]):
            Span = (D90_actual - D10_actual) / D50_actual
            inputs["Span"] = Span
        else:
            inputs["Span"] = "Not Available"
        print(f"Powder: D10={D10_actual}, D50={D50_actual}, D90={D90_actual}, Span={inputs['Span']}")
        
        # Ask user for output filename
        output_filename = input("Enter output Excel filename: ").strip()
        if output_filename == "":
            output_filename = "output.xlsx"
        # Ensure it has .xlsx extension
        if not output_filename.lower().endswith('.xlsx'):
            output_filename += '.xlsx'
        inputs["output_filename"] = output_filename
        
        return inputs
    except Exception as e:
        print(f"Error in input collection: {e}")
        return None