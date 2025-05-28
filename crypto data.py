# Cryptocurrency dataset
crypto_data = [
    {
        'symbol': 'BTC',
        'name': 'Bitcoin',
        'price_trend': 'increasing',
        'energy_consumption': 'high',
        'project_viability': 'high',
        'market_cap': 'high'
    },
    {
        'symbol': 'ETH',
        'name': 'Ethereum',
        'price_trend': 'stable',
        'energy_consumption': 'medium',
        'project_viability': 'high',
        'market_cap': 'high'
    },
    {
        'symbol': 'SOL',
        'name': 'Solana',
        'price_trend': 'increasing',
        'energy_consumption': 'low',
        'project_viability': 'medium',
        'market_cap': 'medium'
    },
    {
        'symbol': 'ADA',
        'name': 'Cardano',
        'price_trend': 'decreasing',
        'energy_consumption': 'low',
        'project_viability': 'high',
        'market_cap': 'medium'
    }
]

def get_crypto_info(symbol):
    """Retrieve cryptocurrency data by symbol"""
    symbol = symbol.upper()
    for crypto in crypto_data:
        if crypto['symbol'] == symbol:
            return crypto
    return None

def analyze_profitability(crypto):
    """Evaluate investment profitability potential"""
    if crypto['price_trend'] == 'increasing':
        return "Strong positive trend"
    elif crypto['price_trend'] == 'stable':
        return "Neutral trend"
    return "Declining trend - caution advised"

def analyze_sustainability(crypto):
    """Evaluate environmental and project sustainability"""
    sustainability_score = 0
    
    # Energy efficiency scoring
    if crypto['energy_consumption'] == 'low':
        sustainability_score += 2
    elif crypto['energy_consumption'] == 'medium':
        sustainability_score += 1
    
    # Project viability scoring
    if crypto['project_viability'] == 'high':
        sustainability_score += 2
    elif crypto['project_viability'] == 'medium':
        sustainability_score += 1
    
    # Generate recommendation
    if sustainability_score >= 3:
        return "Excellent sustainability"
    elif sustainability_score >= 2:
        return "Good sustainability"
    return "Questionable long-term sustainability"

def generate_advice(crypto):
    """Generate investment advice based on analysis"""
    profitability = analyze_profitability(crypto)
    sustainability = analyze_sustainability(crypto)
    
    # Core decision rules
    if "Strong positive" in profitability and "Excellent" in sustainability:
        return "STRONG BUY: Excellent fundamentals and momentum"
    
    if "Declining" in profitability:
        if "Excellent" in sustainability:
            return "HOLD: Poor momentum but strong fundamentals"
        return "SELL: Weak momentum and sustainability concerns"
    
    if "Strong positive" in profitability:
        return "BUY: Strong momentum but evaluate sustainability"
    
    if "Excellent" in sustainability:
        return "ACCUMULATE: Strong fundamentals with neutral momentum"
    
    return "NEUTRAL: Monitor for changes in fundamentals or momentum"

def chatbot():
    """Main chatbot function"""
    print("🤖 Crypto Advisor: Hi! I analyze cryptocurrencies for investment potential.")
    print("Available coins: BTC, ETH, SOL, ADA\nType 'exit' to quit")
    
    while True:
        user_input = input("\nYou: ").strip().upper()
        
        if user_input in ['EXIT', 'QUIT']:
            print("🤖 Crypto Advisor: Goodbye! Always DYOR (Do Your Own Research).")
            break
        
        crypto = get_crypto_info(user_input)
        
        if not crypto:
            print("🤖 Crypto Advisor: Coin not found. Try BTC, ETH, SOL, or ADA")
            continue
        
        print(f"\n🤖 Crypto Advisor: Analyzing {crypto['name']} ({crypto['symbol']})...")
        print(f"• Price Trend: {crypto['price_trend'].capitalize()}")
        print(f"• Energy Efficiency: {crypto['energy_consumption'].capitalize()}")
        print(f"• Project Viability: {crypto['project_viability'].capitalize()}")
        
        print(f"\nProfitability Analysis: {analyze_profitability(crypto)}")
        print(f"Sustainability Analysis: {analyze_sustainability(crypto)}")
        
        print(f"\n💡 Investment Advice: {generate_advice(crypto)}")
        print("\nAnalyze another coin or type 'exit'")

# Start the chatbot
if __name__ == "__main__":
    chatbot()