import numpy as np

def classify_fault(h2, ch4, c2h2, c2h4, c2h6, co, co2):
    try:
        ratio1 = ch4 / h2 if h2 != 0 else 0
        ratio2 = c2h2 / c2h4 if c2h4 != 0 else 0
        ratio3 = c2h4 / c2h6 if c2h6 != 0 else 0

        if ratio1 < 0.1 and ratio2 < 0.5 and ratio3 < 1:
            fault = "PD (Partial Discharge)"
        elif ratio1 < 1 and ratio2 < 0.75 and ratio3 > 1:
            fault = "D1 (Low Energy Discharge)"
        elif ratio1 > 1 and ratio2 > 1 and ratio3 > 3:
            fault = "D2 (High Energy Discharge)"
        elif ratio1 > 0.1 and ratio2 < 0.5 and 1 < ratio3 < 3:
            fault = "T1 (Thermal Fault <300°C)"
        elif ratio1 > 0.1 and 0.5 <= ratio2 <= 1 and 3 <= ratio3 < 10:
            fault = "T2 (Thermal Fault 300–700°C)"
        elif ratio3 >= 10:
            fault = "T3 (Thermal Fault >700°C)"
        else:
            fault = "Unclassified / Normal"

        explanation = f"Ratios (CH4/H2={ratio1:.2f}, C2H2/C2H4={ratio2:.2f}, C2H4/C2H6={ratio3:.2f})"
        return fault, explanation
    except Exception as e:
        return "Error", str(e)
