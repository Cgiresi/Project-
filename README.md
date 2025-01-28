def calculate_dcf(cash_flows, discount_rate):
    """
    This function calculates the Discounted Cash Flow (DCF) of a series of cash flows.
    :param cash_flows: List of expected cash flows (e.g., [1000, 1500, 2000, 2500, 3000])
    :param discount_rate: The discount rate as a decimal (e.g., 0.1 for 10%)
    :return: The calculated DCF value
    """
    dcf_value = 0
    for i, cash_flow in enumerate(cash_flows):
        # Calculate the present value of each cash flow
        present_value = cash_flow / ((1 + discount_rate) ** (i + 1))
        dcf_value += present_value
    return dcf_value

# Example usage
cash_flows = [1000, 1500, 2000, 2500, 3000]  # Expected cash flows for 5 years
discount_rate = 0.1  # Discount rate (10%)

dcf_value = calculate_dcf(cash_flows, discount_rate)
print(f"The calculated DCF value is: ${dcf_value:.2f}")
