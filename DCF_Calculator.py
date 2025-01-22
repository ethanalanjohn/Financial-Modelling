ebit = 179	 
tax_rate = 0.25  # Corporate tax rate (25%)
growth_rate = 0.03  # Growth rate (3%)
roc_new_investments = 0.15  # Return on Capital for new investments (15%)
wacc =0.146  # Weighted Average Cost of Capital (8%)
n = 1 # Number of years to discount back

depreciation = 150000  # Depreciation expense
change_in_nwc = 50000  # Change in Net Working Capital
capex = 200000  # Capital Expenditure
change_in_op_provisions = 25000  # Change in Operating Provisions

re =0.1   # Cost of equity (8%)
E = 500000  # Market value of equity (in currency, e.g., dollars)
rd = 0.05  # Cost of debt (5%)
tc = 0.3   # Corporate tax rate (30%)
D = 200000  # Market value of debt (in currency)



def calculate_wacc(re, E, rd, tc, D):
   
    V = E + D  # Total value of the firm (Equity + Debt)
    wacc = (re * (E / V)) + (rd * (1 - tc) * (D / V))
    return wacc

# Example usage:


wacc = calculate_wacc(re, E, rd, tc, D)
print(f"The WACC is: {wacc:.2%}")
def calculate_fcf(ebit, tc, depreciation, change_in_nwc, capex, change_in_op_provisions):
    """
    Calculate Free Cash Flow (FCF).

    Parameters:
    ebit (float): Earnings Before Interest and Taxes (EBIT).
    tc (float): Corporate tax rate (as a decimal, e.g., 0.3 for 30%).
    depreciation (float): Depreciation and amortization expense.
    change_in_nwc (float): Change in Net Working Capital.
    capex (float): Capital Expenditure.
    change_in_op_provisions (float): Change in Operating Provisions.

    Returns:
    float: Free Cash Flow (FCF).
    """
    # Operating profit after tax
    op_after_tax = ebit * (1 - tc)
    
    # FCF formula
    fcf = (
        op_after_tax +  # EBIT * (1 - tc)
        depreciation -  # Add back depreciation
        change_in_nwc -  # Subtract change in Net Working Capital
        capex +  # Subtract capital expenditure
        change_in_op_provisions  # Add change in operating provisions
    )
    
    return fcf

fcf = calculate_fcf(ebit, tc, depreciation, change_in_nwc, capex, change_in_op_provisions)
print(f"Free Cash Flow (FCF): {fcf}")
def calculate_terminal_value(ebit, tax_rate, growth_rate, roc_new_investments, wacc):
    """
    Calculates the Terminal Value using the formula:
    (EBIT * (1 - tc) * (1 + g) * (1 - (g / ROCB))) / (WACC - g)
    """
    # Calculate NOPAT
    nopat = ebit * (1 - tax_rate)

    # Terminal Value formula
    terminal_value = (nopat * (1+growth_rate) * (1 - (growth_rate / roc_new_investments))) / (wacc - growth_rate)
    
    return terminal_value
def calculate_pv_terminal_value(terminal_value, wacc, n):
    """
    Calculates the Present Value of the Terminal Value using the formula:
    PV(Terminal Value) = Terminal Value / (1 + WACC)^n
    """
    pv_terminal_value = terminal_value / ((1 + wacc) ** n)
    return pv_terminal_value
# Calculate Terminal Value and its Present Value
terminal_value = calculate_terminal_value(ebit, tax_rate, growth_rate, roc_new_investments, wacc)
print(f"Terminal Value: {terminal_value:,.2f}")
pv_terminal_value = calculate_pv_terminal_value(terminal_value, wacc, n)
print(f"Present Value of Terminal Value: {pv_terminal_value:,.2f}")
