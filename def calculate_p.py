def calculate_discount(price, discount_percent):
    
    if discount_percent >= 20:
        discount_amount = price * (discount_percent / 100)
        return price - discount_amount
    else:
        return price

# Prompt the user for input
try:
    original_price = float(input("Enter the original price of the item:10"))
    discount_percentage = float(input("Enter the discount percentage: "))
    
    final_price = calculate_discount(original_price, discount_percentage)
    
    if discount_percentage >= 20:
        print(f"Discount applied. Final price: ${final_price:.2f}")
    else:
        print(f"No discount applied. Final price: ${final_price:.2f}")
except ValueError:
    print("Invalid input. Please enter numeric values.")
