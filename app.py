"""
🤖 Dividend Intelligence Agent (DIA)
A comprehensive AI agent for dividend event analysis and prediction
Deployed on Streamlit Cloud
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# MUST be first Streamlit command
st.set_page_config(
    page_title="Dividend Intelligence Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .upcoming-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .event-card {
        background: #f8f9fa;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
        color: black;
        font-weight: bold;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
</style>
""", unsafe_allow_html=True)

class Market(Enum):
    US = "🇺🇸 United States"
    UK = "🇬🇧 United Kingdom"
    DE = "🇩🇪 Germany"
    FR = "🇫🇷 France"
    JP = "🇯🇵 Japan"
    CA = "🇨🇦 Canada"
    AU = "🇦🇺 Australia"
    CH = "🇨🇭 Switzerland"
    NL = "🇳🇱 Netherlands"
    SE = "🇸🇪 Sweden"

@dataclass
class DividendEvent:
    ticker: str
    company_name: str
    sector: str
    industry: str
    market: str
    ex_date: datetime
    payment_date: Optional[datetime]
    record_date: Optional[datetime]
    dividend_amount: float
    dividend_yield: float
    payout_ratio: Optional[float]
    frequency: str
    country: str
    currency: str
    five_days_before: pd.DataFrame
    five_days_after: pd.DataFrame
    price_change_before: float
    price_change_after: float
    volume_surge: float
    volatility_change: float
    market_cap: Optional[float]
    pe_ratio: Optional[float]
    beta: Optional[float]
    is_upcoming: bool = False
    days_until: Optional[int] = None

@st.cache_resource
class DividendIntelligenceAgent:
    def __init__(self):
        self.universe = self._build_universe()
        self._cache = {}
        
    def _build_universe(self) -> Dict[Market, List[str]]:
        """Build comprehensive stock universe by market"""
        return {
            Market.US: [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'BRK-B', 'TSLA', 'V', 'JPM',
                'JNJ', 'WMT', 'UNH', 'MA', 'PG', 'HD', 'CVX', 'MRK', 'ABBV', 'PEP',
                'KO', 'BAC', 'PFE', 'AVGO', 'COST', 'TMO', 'DIS', 'ABT', 'ADBE', 'CRM',
                'ACN', 'VZ', 'DHR', 'NKE', 'TXN', 'NEE', 'PM', 'RTX', 'HON', 'QCOM',
                'AMGN', 'UPS', 'LOW', 'MS', 'UNP', 'SBUX', 'CAT', 'IBM', 'GE', 'CVS'
            ],
            Market.UK: [
                'AZN', 'SHEL', 'UL', 'BP', 'RIO', 'GSK', 'BARC', 'LLOY', 'VOD', 'AAL',
                'RELX', 'NG', 'LSEG', 'EXPN', 'CRH', 'SN', 'SMT', 'HLN', 'BATS', 'RKT',
                'CNA', 'TSCO', 'BKG', 'IMB', 'ANTO', 'BLND', 'PSN', 'AHT', 'CRDA', 'ENT'
            ],
            Market.DE: [
                'SAP', 'SIE.DE', 'ALV.DE', 'DTE.DE', 'BAS.DE', 'BAYN.DE', 'DAI.DE', 
                'BMW.DE', 'VOW3.DE', 'ADS.DE', 'MRK.DE', 'FRE.DE', 'HEN3.DE', 'BEI.DE',
                'CON.DE', 'DBK.DE', 'DPW.DE', 'EOAN.DE', 'FME.DE', 'HEI.DE'
            ],
            Market.FR: [
                'MC.PA', 'OR.PA', 'SAN.PA', 'AIR.PA', 'BNP.PA', 'TTE.PA', 'CS.PA', 
                'DG.PA', 'VIV.PA', 'CAP.PA', 'DSY.PA', 'SU.PA', 'AI.PA', 'EL.PA',
                'KER.PA', 'RI.PA', 'PUB.PA', 'WLN.PA', 'VIE.PA', 'GLE.PA'
            ],
            Market.JP: [
                '7203.T', '6758.T', '9984.T', '8306.T', '9432.T', '9433.T', '9983.T',
                '6861.T', '9434.T', '8316.T', '8058.T', '8766.T', '7267.T', '8411.T',
                '9020.T', '9437.T', '4502.T', '8308.T', '8031.T', '7751.T'
            ],
            Market.CA: [
                'RY.TO', 'TD.TO', 'ENB.TO', 'SHOP.TO', 'BNS.TO', 'BMO.TO', 'CM.TO',
                'CNR.TO', 'TRP.TO', 'SU.TO', 'MFC', 'BCE', 'T.TO', 'NA.TO',
                'CP.TO', 'WCN.TO', 'CSU.TO', 'ATD.TO', 'FNV.TO', 'PPL.TO'
            ],
            Market.AU: [
                'CBA.AX', 'BHP.AX', 'NAB.AX', 'WBC.AX', 'ANZ.AX', 'WES.AX', 'MQG.AX',
                'GMG.AX', 'CSL.AX', 'RIO.AX', 'FMG.AX', 'WDS.AX', 'TLS.AX', 'WOW.AX',
                'COL.AX', 'JHX.AX', 'REA.AX', 'ALL.AX', 'QBE.AX', 'TCL.AX'
            ],
            Market.CH: [
                'NESN.SW', 'ROG.SW', 'NOVN.SW', 'UBSG.SW', 'ABBN.SW', 'ZURN.SW',
                'LONN.SW', 'GIVN.SW', 'SGSN.SW', 'SIKA.SW', 'SCMN.SW', 'UHR.SW',
                'OERL.SW', 'CFR.SW', 'TEMN.SW', 'SOON.SW', 'SLHN.SW', 'GEBN.SW'
            ],
            Market.NL: [
                'ASML', 'UNA.AS', 'ING', 'ADYEN.AS', 'DSM.AS', 'AKZA.AS', 'HEIA.AS',
                'ASRNL.AS', 'NN.AS', 'REN.AS', 'MT.AS', 'PHIA.AS', 'KPN.AS', 'ABN.AS'
            ],
            Market.SE: [
                'ERIC-B.ST', 'ESSITY-B.ST', 'HM-B.ST', 'INVE-B.ST', 'NDA-SE.ST',
                'SAND.ST', 'SCA-B.ST', 'SEB-A.ST', 'SHB-A.ST', 'SKF-B.ST',
                'SWED-A.ST', 'TEL2-B.ST', 'TELIA.ST', 'VOLV-B.ST', 'ATCO-A.ST'
            ]
        }
    
    def get_market_stocks(self, market: Market) -> List[str]:
        """Get stocks for selected market"""
        return self.universe.get(market, self.universe[Market.US])
    
    @st.cache_data(ttl=3600)
    def fetch_comprehensive_data(_self, ticker: str) -> Optional[Dict]:
        """Fetch comprehensive stock data with caching"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            history = stock.history(period="10y")
            dividends = stock.dividends
            
            if history.empty:
                return None
            
            return {
                'info': info,
                'history': history,
                'dividends': dividends,
                'ticker': ticker
            }
        except Exception as e:
            return None
    
    def analyze_dividend_event(self, ticker: str, ex_date: datetime, 
                              amount: float, data: Dict) -> DividendEvent:
        """Analyze a specific dividend event with 5-day windows"""
        history = data['history']
        info = data['info']
        
        # Get price windows
        before_start = ex_date - timedelta(days=10)
        before_end = ex_date
        after_start = ex_date
        after_end = ex_date + timedelta(days=10)
        
        mask_before = (history.index >= before_start) & (history.index <= before_end)
        mask_after = (history.index >= after_start) & (history.index <= after_end)
        
        before_df = history.loc[mask_before].tail(5)
        after_df = history.loc[mask_after].head(5)
        
        # Calculate metrics
        price_change_before = 0
        price_change_after = 0
        volume_surge = 0
        volatility_change = 0
        
        if len(before_df) >= 2:
            start_p = before_df['Close'].iloc[0]
            end_p = before_df['Close'].iloc[-1]
            price_change_before = ((end_p - start_p) / start_p) * 100
            
        if len(after_df) >= 2:
            start_p = after_df['Close'].iloc[0]
            end_p = after_df['Close'].iloc[-1]
            price_change_after = ((end_p - start_p) / start_p) * 100
            
        if not before_df.empty and not after_df.empty:
            vol_before = before_df['Volume'].mean()
            vol_after = after_df['Volume'].mean()
            if vol_before > 0:
                volume_surge = ((vol_after - vol_before) / vol_before) * 100
            
            if len(before_df) > 1 and len(after_df) > 1:
                vol_before = before_df['Close'].pct_change().std() * np.sqrt(252) * 100
                vol_after = after_df['Close'].pct_change().std() * np.sqrt(252) * 100
                volatility_change = vol_after - vol_before
        
        is_upcoming = ex_date > datetime.now()
        days_until = (ex_date - datetime.now()).days if is_upcoming else None
        
        return DividendEvent(
            ticker=ticker,
            company_name=info.get('longName', ticker),
            sector=info.get('sector', 'Unknown'),
            industry=info.get('industry', 'Unknown'),
            market=info.get('exchange', 'Unknown'),
            ex_date=ex_date,
            payment_date=ex_date + timedelta(days=30) if not pd.isna(ex_date) else None,
            record_date=None,
            dividend_amount=amount,
            dividend_yield=info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0,
            payout_ratio=info.get('payoutRatio'),
            frequency=self._determine_frequency(ticker, data['dividends']),
            country=info.get('country', 'Unknown'),
            currency=info.get('currency', 'USD'),
            five_days_before=before_df,
            five_days_after=after_df,
            price_change_before=price_change_before,
            price_change_after=price_change_after,
            volume_surge=volume_surge,
            volatility_change=volatility_change,
            market_cap=info.get('marketCap'),
            pe_ratio=info.get('trailingPE'),
            beta=info.get('beta'),
            is_upcoming=is_upcoming,
            days_until=days_until
        )
    
    def _determine_frequency(self, ticker: str, dividends: pd.Series) -> str:
        """Determine dividend frequency from history"""
        if len(dividends) < 2:
            return "Unknown"
        
        dates = dividends.index
        gaps = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
        avg_gap = np.mean(gaps)
        
        if avg_gap <= 100:
            return "Quarterly"
        elif avg_gap <= 200:
            return "Semi-Annual"
        else:
            return "Annual"
    
    def scan_market(self, market: Market, years: int = 10, 
                   progress_bar=None, status_text=None) -> List[DividendEvent]:
        """Scan entire market for dividend events"""
        tickers = self.get_market_stocks(market)
        all_events = []
        
        for idx, ticker in enumerate(tickers):
            if progress_bar:
                progress_bar.progress((idx + 1) / len(tickers))
            if status_text:
                status_text.text(f"🔍 Analyzing {ticker} ({idx+1}/{len(tickers)})...")
            
            data = self.fetch_comprehensive_data(ticker)
            if not data or data['dividends'].empty:
                continue
            
            cutoff = datetime.now() - timedelta(days=365 * years)
            recent_divs = data['dividends'][data['dividends'].index >= cutoff]
            
            for date, amount in recent_divs.items():
                try:
                    event = self.analyze_dividend_event(ticker, date.to_pydatetime(), amount, data)
                    all_events.append(event)
                except Exception as e:
                    continue
        
        all_events.sort(key=lambda x: (not x.is_upcoming, x.ex_date), reverse=True)
        return all_events
    
    def get_upcoming_events(self, market: Market, days_ahead: int = 90) -> List[DividendEvent]:
        """Get upcoming dividend events based on historical patterns"""
        tickers = self.get_market_stocks(market)
        upcoming = []
        
        for ticker in tickers:
            data = self.fetch_comprehensive_data(ticker)
            if not data:
                continue
            
            info = data['info']
            divs = data['dividends']
            
            if divs.empty:
                continue
            
            last_date = divs.index[-1]
            last_amount = divs.iloc[-1]
            frequency = self._determine_frequency(ticker, divs)
            
            if frequency == "Quarterly":
                next_date = last_date + timedelta(days=90)
            elif frequency == "Semi-Annual":
                next_date = last_date + timedelta(days=180)
            else:
                next_date = last_date + timedelta(days=365)
            
            if next_date > datetime.now() and (next_date - datetime.now()).days <= days_ahead:
                try:
                    event = self.analyze_dividend_event(ticker, next_date.to_pydatetime(), last_amount, data)
                    upcoming.append(event)
                except:
                    continue
        
        upcoming.sort(key=lambda x: x.ex_date)
        return upcoming

def render_event_card(event: DividendEvent, detailed: bool = False):
    """Render a dividend event card"""
    if event.is_upcoming:
        st.markdown(f"""
        <div class="upcoming-card">
            <h4>📅 {event.ticker} - {event.company_name[:30]}</h4>
            <p><strong>Ex-Date:</strong> {event.ex_date.strftime('%Y-%m-%d')} (in {event.days_until} days)</p>
            <p><strong>Dividend:</strong> ${event.dividend_amount:.2f} ({event.frequency}) | Yield: {event.dividend_yield:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        with col1:
            st.write(f"**{event.ticker}** - {event.company_name[:25]}")
            st.caption(f"{event.sector} | {event.ex_date.strftime('%Y-%m-%d')}")
        with col2:
            st.metric("Div", f"${event.dividend_amount:.2f}")
        with col3:
            delta_color = "normal" if event.price_change_after >= 0 else "inverse"
            st.metric("Impact", f"{event.price_change_after:+.1f}%", delta_color=delta_color)
        with col4:
            st.metric("Volume", f"{event.volume_surge:+.0f}%")
        
        if detailed and not event.five_days_before.empty:
            with st.expander("📊 Detailed 5-Day Analysis"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**5 Days Before Ex-Date**")
                    display_df = event.five_days_before[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
                    display_df = display_df.round(2)
                    st.dataframe(display_df, use_container_width=True)
                with col2:
                    st.write("**5 Days After Ex-Date**")
                    display_df = event.five_days_after[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
                    display_df = display_df.round(2)
                    st.dataframe(display_df, use_container_width=True)
                
                # Price chart
                fig = go.Figure()
                combined = pd.concat([event.five_days_before, event.five_days_after])
                fig.add_trace(go.Scatter(
                    x=combined.index, 
                    y=combined['Close'], 
                    mode='lines+markers',
                    name='Close Price',
                    line=dict(color='#00C9FF', width=3)
                ))
                fig.add_vline(
                    x=event.ex_date, 
                    line_dash="dash", 
                    line_color="red",
                    annotation_text="Ex-Date",
                    annotation_position="top"
                )
                fig.update_layout(
                    title=f"{event.ticker} Price Action Around Dividend",
                    height=400,
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 Dividend Intelligence Agent</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Dividend Event Analysis & Prediction</p>', unsafe_allow_html=True)
    
    # Initialize
    if 'agent' not in st.session_state:
        st.session_state.agent = DividendIntelligenceAgent()
    if 'events' not in st.session_state:
        st.session_state.events = []
    if 'selected_market' not in st.session_state:
        st.session_state.selected_market = Market.US
    
    agent = st.session_state.agent
    
    # Sidebar
    with st.sidebar:
        st.title("🎛️ Control Panel")
        
        market = st.selectbox(
            "🌍 Select Market",
            options=list(Market),
            format_func=lambda x: x.value,
            index=list(Market).index(st.session_state.selected_market)
        )
        st.session_state.selected_market = market
        
        st.divider()
        
        mode = st.radio(
            "📊 Analysis Mode",
            ["Dashboard", "Market Scanner", "Stock Lookup"]
        )
        
        st.divider()
        
        st.info("💡 **Tip:** Use Market Scanner for 10-year historical analysis")
        
        if st.button("🗑️ Clear Cache"):
            st.cache_data.clear()
            st.success("Cache cleared!")
    
    # Main Content
    if mode == "Dashboard":
        render_dashboard(agent, market)
    elif mode == "Market Scanner":
        render_market_scanner(agent, market)
    else:
        render_stock_lookup(agent, market)

def render_dashboard(agent: DividendIntelligenceAgent, market: Market):
    """Render main dashboard"""
    st.header(f"{market.value} Dashboard")
    
    # Stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Stocks Tracked", len(agent.get_market_stocks(market)))
    with col2:
        st.metric("Analysis Period", "10 Years")
    with col3:
        st.metric("Window Size", "±5 Days")
    with col4:
        st.metric("Data Source", "Yahoo Finance")
    
    # Upcoming Events
    st.subheader("📅 Upcoming Dividend Events (Next 30 Days)")
    
    with st.spinner("Loading upcoming events..."):
        upcoming = agent.get_upcoming_events(market, days_ahead=30)
    
    if upcoming:
        cols = st.columns(min(3, len(upcoming)))
        for idx, event in enumerate(upcoming[:6]):
            with cols[idx % 3]:
                render_event_card(event)
    else:
        st.info("No upcoming dividend events found in the next 30 days")
    
    # Quick Actions
    st.subheader("🚀 Quick Actions")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔍 Scan Full Market", type="primary"):
            st.session_state.scan_triggered = True
            st.rerun()
    with col2:
        ticker = st.text_input("Or enter ticker:", "").upper()
        if ticker and st.button("Analyze Stock"):
            st.session_state.quick_ticker = ticker
            st.session_state.mode = "Stock Lookup"
            st.rerun()

def render_market_scanner(agent: DividendIntelligenceAgent, market: Market):
    """Render market scanner"""
    st.header(f"🔍 {market.value} Market Scanner")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        years = st.slider("Historical Years", 1, 10, 5)
    with col2:
        min_div = st.number_input("Min Dividend ($)", 0.0, 10.0, 0.0)
    with col3:
        st.write("")
        st.write("")
        scan_btn = st.button("🚀 Start Deep Scan", type="primary")
    
    if scan_btn or st.session_state.get('scan_triggered'):
        if 'scan_triggered' in st.session_state:
            del st.session_state.scan_triggered
            
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with st.spinner(f"Scanning {len(agent.get_market_stocks(market))} stocks..."):
            events = agent.scan_market(market, years, progress_bar, status_text)
            st.session_state.events = events
        
        st.success(f"✅ Found {len(events)} dividend events")
    
    if st.session_state.events:
        # Filters
        st.subheader("🔎 Filter Results")
        col1, col2, col3 = st.columns(3)
        with col1:
            sectors = sorted(list(set([e.sector for e in st.session_state.events])))
            selected_sectors = st.multiselect("Sectors", sectors, default=sectors[:5])
        with col2:
            sort_opt = st.selectbox("Sort By", ["Date (Newest)", "Dividend Amount", "Yield", "Price Impact"])
        with col3:
            limit = st.number_input("Show Top N", 10, 100, 20)
        
        # Filter and sort
        filtered = [e for e in st.session_state.events if e.sector in selected_sectors]
        if min_div > 0:
            filtered = [e for e in filtered if e.dividend_amount >= min_div]
        
        if sort_opt == "Date (Newest)":
            filtered.sort(key=lambda x: x.ex_date, reverse=True)
        elif sort_opt == "Dividend Amount":
            filtered.sort(key=lambda x: x.dividend_amount, reverse=True)
        elif sort_opt == "Yield":
            filtered.sort(key=lambda x: x.dividend_yield, reverse=True)
        elif sort_opt == "Price Impact":
            filtered.sort(key=lambda x: abs(x.price_change_after), reverse=True)
        
        # Display
        st.subheader(f"📋 Showing {min(len(filtered), limit)} of {len(filtered)} events")
        
        for event in filtered[:limit]:
            render_event_card(event, detailed=True)
            st.divider()

def render_stock_lookup(agent: DividendIntelligenceAgent, market: Market):
    """Render individual stock lookup"""
    st.header("🔎 Individual Stock Analysis")
    
    # Check for quick ticker from dashboard
    default_ticker = st.session_state.get('quick_ticker', 'AAPL')
    if 'quick_ticker' in st.session_state:
        del st.session_state.quick_ticker
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        ticker = st.text_input("Enter Ticker", default_ticker).upper()
        analyze_btn = st.button("Analyze", type="primary")
        
        if analyze_btn:
            with st.spinner(f"Fetching {ticker}..."):
                data = agent.fetch_comprehensive_data(ticker)
                if data:
                    st.session_state.current_stock = data
                else:
                    st.error("Stock not found")
    
    with col2:
        if 'current_stock' in st.session_state:
            data = st.session_state.current_stock
            info = data['info']
            
            st.subheader(f"{info.get('longName', ticker)} ({ticker})")
            st.caption(f"{info.get('sector', 'Unknown')} | {info.get('industry', 'Unknown')}")
            
            # Metrics
            cols = st.columns(4)
            metrics = [
                ("Market Cap", f"${info.get('marketCap', 0)/1e9:.1f}B" if info.get('marketCap') else "N/A"),
                ("P/E Ratio", f"{info.get('trailingPE', 'N/A')}" if info.get('trailingPE') else "N/A"),
                ("Div Yield", f"{info.get('dividendYield', 0)*100:.2f}%" if info.get('dividendYield') else "0%"),
                ("Beta", f"{info.get('beta', 'N/A')}" if info.get('beta') else "N/A")
            ]
            for col, (label, value) in zip(cols, metrics):
                col.metric(label, value)
            
            # Dividend History
            if not data['dividends'].empty:
                st.subheader("📈 10-Year Dividend History")
                
                div_df = data['dividends'].reset_index()
                div_df.columns = ['Date', 'Dividend']
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=div_df['Date'], 
                    y=div_df['Dividend'],
                    marker_color='rgb(0, 201, 255)'
                ))
                fig.update_layout(
                    title="Historical Dividend Payments",
                    xaxis_title="Date",
                    yaxis_title="Amount ($)",
                    template="plotly_white",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Recent events
                st.subheader("🎯 Recent Event Analysis")
                cutoff = datetime.now() - timedelta(days=365*3)
                recent_divs = data['dividends'][data['dividends'].index >= cutoff]
                
                for date, amount in list(recent_divs.items())[-5:]:
                    event = agent.analyze_dividend_event(ticker, date.to_pydatetime(), amount, data)
                    render_event_card(event, detailed=True)
            else:
                st.warning("No dividend history available")

if __name__ == "__main__":
    main()
