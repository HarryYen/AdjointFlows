def compute_smoothing_sigmas(T_min, v_avg=4.5):
    """
    Compute Gaussian smoothing sigmas (horizontal and vertical) for adjoint FWT.
    """
    import numpy as np
    C_h = 0.5  # horizontal constant
    C_v = 0.2  # vertical constant

    Gamma_h = C_h * T_min * v_avg
    Gamma_v = C_v * T_min * v_avg

    sigma_h = Gamma_h / np.sqrt(8)
    sigma_v = Gamma_v / np.sqrt(8)

    return sigma_h, sigma_v, Gamma_h, Gamma_v

def smoothing_sigma_script():
    print("Adjoint FWT Smoothing Parameter Calculator")
    print("-------------------------------------------")
    
    try:
        # Get user input
        T_min = float(input("Enter the shortest period (T_min) in seconds: "))
        v_avg_input = input("Enter average wave speed (in km/s) [press Enter to use default 4.5 km/s]: ")

        # Default wave speed
        v_avg = float(v_avg_input) if v_avg_input.strip() else 4.5

        # Compute sigmas and scalelengths
        sigma_h, sigma_v, Gamma_h, Gamma_v = compute_smoothing_sigmas(T_min, v_avg)

        # Display results
        print("\nComputed Smoothing Parameters:")
        print(f"  Horizontal scalelength Gamma_h = {Gamma_h:.2f} km")
        print(f"  Vertical scalelength Gamma_v = {Gamma_v:.2f} km")
        print(f"  Horizontal Gaussian sigma_h = {sigma_h:.2f} km")
        print(f"  Vertical Gaussian sigma_v = {sigma_v:.2f} km")

    except ValueError:
        print("Invalid input. Please enter numeric values for period and wave speed.")

if __name__ == "__main__":
    smoothing_sigma_script()
