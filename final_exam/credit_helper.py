
import QuantLib as ql
import numpy as np
import pandas as pd
import datetime as dt
from scipy.optimize import minimize
from scipy.stats import norm
from scipy import optimize



import matplotlib.pyplot as plt


def get_ql_date(date) -> ql.Date:
    """
    convert dt.date to ql.Date
    """
    if isinstance(date, dt.date):
        return ql.Date(date.day, date.month, date.year)
    elif isinstance(date, str):
        date = dt.datetime.strptime(date, "%Y-%m-%d").date()
        return ql.Date(date.day, date.month, date.year)
    else:
        raise ValueError(f"to_qldate, {type(date)}, {date}")


def create_schedule_from_symbology(details: dict):
    '''Create a QuantLib cashflow schedule from symbology details dictionary (usually one row of the symbology dataframe)
    '''
    # Create maturity from details['maturity']
    maturity = get_ql_date(details['maturity'])
    
    # Create acc_first from details['acc_first']
    acc_first = get_ql_date(details['acc_first'])
    
    # Create calendar for Corp and Govt asset classes
    calendar = ql.UnitedStates(ql.UnitedStates.GovernmentBond)
    
    # define period from details['cpn_freq'] ... can be hard-coded to 2 = semi-annual frequency
    period = ql.Period(2)
    
    # business_day_convention
    business_day_convention = ql.Unadjusted
    
    # termination_date_convention
    termination_date_convention = ql.Unadjusted
    
    # date_generation
    date_generation=ql.DateGeneration.Backward
    
    # Create schedule using ql.MakeSchedule interface (with keyword arguments)
    schedule = ql.MakeSchedule(effectiveDate=acc_first,  # this may not be the same as the bond's start date
                            terminationDate=maturity,
                            tenor=period,
                            calendar=calendar,
                            convention=business_day_convention,
                            terminalDateConvention=termination_date_convention,
                            rule=date_generation,
                            endOfMonth=True,
                            firstDate=ql.Date(),
                            nextToLastDate=ql.Date())
    return schedule


def create_bond_from_symbology(details: dict):
    '''Create a US fixed rate bond object from symbology details dictionary (usually one row of the symbology dataframe)
    '''
    
     # Create day_count from details['dcc']
     # For US Treasuries use ql.ActualActual(ql.ActualActual.ISMA)
     # For US Corporates use ql.Thirty360(ql.Thirty360.USA)
    if details['dcc'] == '30/360':
        day_count = ql.Thirty360(ql.Thirty360.USA)
    elif details['dcc'] == 'ACT/ACT':
        day_count = ql.ActualActual(ql.ActualActual.ISMA)
    else:
        raise ValueError(f"unsupported day count, {type(details['dcc'])}, {details['dcc']}")
    
    # Create day_count from details['start_date']    
    issue_date = get_ql_date(details['start_date'])
    
    # Create days_settle from details['days_settle']
    days_settle = int(float(details['days_settle']))

    # Create days_settle from details['coupon']
    coupon = float(details['coupon'])/100.

    # Create cashflow schedule
    schedule = create_schedule_from_symbology(details)
    
    face_value = 100
    redemption = 100
    
    payment_convention = ql.Unadjusted
        
    # Create fixed rate bond object
    fixed_rate_bond = ql.FixedRateBond(
        days_settle,
        face_value,
        schedule,
        [coupon],
        day_count,
        payment_convention,
        redemption,
        issue_date)        

    return fixed_rate_bond


def get_bond_cashflows(bond: ql.FixedRateBond, calc_date=ql.Date):
    '''Returns all future cashflows as of calc_date, i.e. with payment dates > calc_date.
    '''    
    day_counter = bond.dayCounter()    
    
    x = [(cf.date(), day_counter.yearFraction(calc_date, cf.date()), cf.amount()) for cf in bond.cashflows()]
    cf_date, cf_yearFrac, cf_amount = zip(*x)
    cashflows_df = pd.DataFrame(data={'CashFlowDate': cf_date, 'CashFlowYearFrac': cf_yearFrac, 'CashFlowAmount': cf_amount})

    # filter for payment dates > calc_date
    cashflows_df = cashflows_df[cashflows_df.CashFlowYearFrac > 0]
    return cashflows_df


def calibrate_yield_curve_from_frame(
        calc_date: ql.Date,
        treasury_details: pd.DataFrame,
        price_quote_column: str):
    '''Create a calibrated yield curve from a details dataframe which includes bid/ask/mid price quotes.
    '''
    ql.Settings.instance().evaluationDate = calc_date

    # Sort dataframe by maturity
    sorted_details_frame = treasury_details.sort_values(by='maturity')    
    
    # For US Treasuries use ql.ActualActual(ql.ActualActual.ISMA)
    day_count = ql.ActualActual(ql.ActualActual.ISMA)

    bond_helpers = []
    
    for index, row in sorted_details_frame.iterrows():
        bond_object = create_bond_from_symbology(row)
        
        tsy_clean_price_quote = row[price_quote_column]
        tsy_clean_price_handle = ql.QuoteHandle(ql.SimpleQuote(tsy_clean_price_quote))
        
        bond_helper = ql.BondHelper(tsy_clean_price_handle, bond_object)
        bond_helpers.append(bond_helper)
        
    yield_curve = ql.PiecewiseLogCubicDiscount(calc_date, bond_helpers, day_count)
    yield_curve.enableExtrapolation()
    return yield_curve


def calibrate_sofr_curve_from_frame(
        calc_date: ql.Date,
        sofr_details: pd.DataFrame,
        rate_quote_column: str):
    '''Create a calibrated yield curve from a SOFR details dataframe which includes rate quotes.
    '''
    ql.Settings.instance().evaluationDate = calc_date

    # Sort dataframe by maturity
    sorted_details_frame = sofr_details.sort_values(by='term')    
    
    # settle_days
    settle_days = 2
    
    # For US SOFR OIS Swaps 
    day_count = ql.Actual360()

    # For US SOFR Swaps     
    calendar = ql.UnitedStates(ql.UnitedStates.GovernmentBond)
    
    sofr_helpers = []
    
    for index, row in sorted_details_frame.iterrows():
        sofr_quote = row[rate_quote_column]
        tenor_in_years = row['term']
        sofr_tenor = ql.Period(tenor_in_years, ql.Years)
        
        # create sofr_rate_helper
        sofr_helper = ql.OISRateHelper(settle_days, sofr_tenor, ql.QuoteHandle(ql.SimpleQuote(sofr_quote/100)), ql.Sofr())
                        
        sofr_helpers.append(sofr_helper)
        
    sofr_yield_curve = ql.PiecewiseLinearZero(settle_days, calendar, sofr_helpers, day_count)
    sofr_yield_curve.enableExtrapolation()
    
    return sofr_yield_curve


def calibrate_cds_hazard_rate_curve(calc_date, sofr_yield_curve_handle, cds_par_spreads_bps, cds_recovery_rate = 0.4):
    '''Calibrate hazard rate curve from CDS Par Spreads'''
    CDS_settle_days = 2

    CDS_day_count = ql.Actual360()

    # CDS standard tenors: 1Y, 2Y, 3Y, 5Y 7Y and 10Y
    CDS_tenors = [ql.Period(y, ql.Years) for y in [1, 2, 3, 5, 7, 10]]
              

    CDS_helpers = [ql.SpreadCdsHelper((cds_par_spread / 10000.0), CDS_tenor, CDS_settle_days, ql.TARGET(),
                                  ql.Quarterly, ql.Following, ql.DateGeneration.TwentiethIMM, CDS_day_count, cds_recovery_rate, sofr_yield_curve_handle)
               
    for (cds_par_spread, CDS_tenor) in zip(cds_par_spreads_bps, CDS_tenors)]

    # bootstrap hazard_rate_curve
    hazard_rate_curve = ql.PiecewiseFlatHazardRate(calc_date, CDS_helpers, CDS_day_count)
    hazard_rate_curve.enableExtrapolation()

    return(hazard_rate_curve)

# Calculate initial term and current time-to-maturity for each bond issue
def get_symbology(df: pd.DataFrame):
    for index, row in df.iterrows():
        start_date = get_ql_date(row['start_date'])
        maturity_date = get_ql_date(row['maturity'])
        today_date = ql.Date(14,4,2023)
        calendar = ql.UnitedStates(ql.UnitedStates.GovernmentBond)
        #set dcc as Actual/365.25
        dcc = ql.Actual36525()
        initial_term = dcc.yearFraction(start_date, maturity_date)
        current_time_to_maturity = dcc.yearFraction(today_date, maturity_date)
        df.at[index, 'term'] = initial_term
        df.at[index, 'TTM'] = current_time_to_maturity
    
    df['term'] = round(df['term'],2)

    return df

def calc_clean_price_with_zspread(fixed_rate_bond, yield_curve_handle, zspread):
    zspread_quote = ql.SimpleQuote(zspread)
    zspread_quote_handle = ql.QuoteHandle(zspread_quote)
    yield_curve_bumped = ql.ZeroSpreadedTermStructure(yield_curve_handle, zspread_quote_handle, ql.Compounded, ql.Semiannual)
    yield_curve_bumped_handle = ql.YieldTermStructureHandle(yield_curve_bumped)
    
    # Set Valuation engine
    bond_engine = ql.DiscountingBondEngine(yield_curve_bumped_handle)
    fixed_rate_bond.setPricingEngine(bond_engine)
    bond_clean_price = fixed_rate_bond.cleanPrice()
    return bond_clean_price

def get_interp_tsy_yield(corp_symbology: pd.DataFrame, otr: pd.DataFrame):
    corp_symbology['interp_tsy_yield'] = 0
    for i in range(len(corp_symbology)):
        if corp_symbology['TTM'][i] <= otr['TTM'][0]:
            corp_symbology['interp_tsy_yield'][i] = otr['mid_yield'][0]
        elif corp_symbology['TTM'][i] >= otr['TTM'][len(otr)-1]:
            corp_symbology['interp_tsy_yield'][i] = otr['mid_yield'][len(otr)-1]
        else:
            ttm_temp = corp_symbology['TTM'][i]
            for j in range(len(otr)):
                if otr['TTM'][j] == ttm_temp:
                    corp_symbology['interp_tsy_yield'][i] = otr['mid_yield'][j]
                elif otr['TTM'][j] > ttm_temp:
                    corp_symbology['interp_tsy_yield'][i] = otr['mid_yield'][j-1] + (ttm_temp - otr['TTM'][j-1])*(otr['mid_yield'][j] - otr['mid_yield'][j-1])/(otr['TTM'][j] - otr['TTM'][j-1])
                    break
    return corp_symbology

# Create a plot using a QuantLib PiecewiseLogCubicDiscount curve
def plot_yield_curve(mid, start_date: ql.Date, end_date: ql.Date, day_count: ql.DayCounter,label: str, CurveType="yield"):
    '''Plot a yield curve from start_date to end_date using a QuantLib PiecewiseLogCubicDiscount curve.
    '''
    # Create a list of dates from start_date to end_date
    dates = [start_date + ql.Period(i, ql.Years) for i in range(0, (end_date.year() - start_date.year())+1)]
    
    #Create another list of dates from start_date to end_date with a 6 month frequency
    dates_6m = [start_date + ql.Period(i, ql.Months) for i in range(0, (end_date.year() - start_date.year())*12+1, 6)]
    
    # Create a list of discount factors from the yield curve
    discount_factors_mid = [mid.discount(d) for d in dates]
    discount_factors_mid_6m = [mid.discount(d) for d in dates_6m]    
    
    # Create a list of mid yields from the yield curve for each date
    mid_yields = [mid.zeroRate(d, day_count, ql.Continuous).rate() for d in dates]
    
    # Create a dataframe of dates, discount factors and zero rates
    df = pd.DataFrame(list(zip(dates, discount_factors_mid, mid_yields)), columns=['Date', 'MidDiscountFactors', 'MidYields'])
    
    #Convert Quantlib dates to Python dates
    df['Date'] = df['Date'].apply(lambda x: x.to_date())
    
    
    if CurveType == "yield":
        # Plot the zero rates
        plt.figure(figsize=(15, 6))
        plt.plot(df['Date'], df['MidYields'], label="mid_yield")
        #, show each year on the x-axis
        plt.xticks(df['Date'], df['Date'].apply(lambda x: x.strftime('%Y')), rotation=45)
        plt.title('Mid Yields')
        plt.xlabel('Date')
        plt.ylabel('Mid Yields')
        plt.legend()
        plt.show()
        return df
    elif CurveType == "discount":
        # Plot the discount factors using a 6 month discretization
        plt.figure(figsize=(15, 6))
        plt.plot(df['Date'], df['DiscountFactor'], label=label)
        #extract the year from the date and show it on the x-axis
        plt.xticks(df['Date'], df['Date'].apply(lambda x: x.strftime('%Y')), rotation=45)
        plt.title('Discount Factors')
        plt.xlabel('Date')
        plt.ylabel('Discount Factor')
        plt.legend()
        plt.show()
        return df
    elif CurveType == "discount_6m":
        df_6m = pd.DataFrame(list(zip(dates_6m, discount_factors_mid_6m)), columns=['Date', 'MidDiscountFactors'])
        df_6m['Date'] = df_6m['Date'].apply(lambda x: x.to_date())
        # Plot the discount factors using a 6 month discretization
        plt.figure(figsize=(15, 8))
        plt.plot(df_6m['Date'], df_6m['MidDiscountFactors'], label="mid_discount")
        plt.xticks(df_6m['Date'], df_6m['Date'].apply(lambda x: x.strftime('%b %Y')), rotation=45)
        plt.title('Discount Factors')
        plt.xlabel('Date')
        plt.ylabel('Discount Factor')
        plt.legend()
        plt.show()
        return df_6m


def calc_bond_metrics(symbology_df: pd.DataFrame, calc_date: ql.Date, model="flat", coupon_freq=ql.Semiannual, yc=None):
    
    ql.Settings.instance().evaluationDate = calc_date
    
    sorted_details_frame = symbology_df.copy()
    flat_rate = ql.SimpleQuote(0.05)
    compounding = ql.Compounded 
    
    for index, row in sorted_details_frame.iterrows():
        fixed_rate_bond = create_bond_from_symbology(row)
        if model == "flat":
            flat_int_rate = ql.InterestRate(flat_rate.value(), fixed_rate_bond.dayCounter(), compounding, coupon_freq)
            bond_duration = ql.BondFunctions.duration(fixed_rate_bond, flat_int_rate)
            bond_convexity = ql.BondFunctions.convexity(fixed_rate_bond, flat_int_rate)
            dv01 = row['mid_dirty'] * bond_duration
            sorted_details_frame.loc[index, 'dv01'] = dv01/100
            sorted_details_frame.loc[index, 'duration'] = bond_duration/100
            sorted_details_frame.loc[index, 'convexity'] = bond_convexity
        elif model == "yc":
            interest_rate_bump = ql.SimpleQuote(0.0)
            flat_yield_curve_bumped = ql.ZeroSpreadedTermStructure(yc, ql.QuoteHandle(interest_rate_bump))
            bond_engine = ql.DiscountingBondEngine(ql.YieldTermStructureHandle(flat_yield_curve_bumped))
            fixed_rate_bond.setPricingEngine(bond_engine)
            price_base = fixed_rate_bond.NPV()
            #-1 bp change in yield
            interest_rate_bump.setValue(-0.0001)
            price_base_1bp = fixed_rate_bond.NPV()
            dv01 = (price_base_1bp - price_base) * 100
            bond_duration = dv01 / row['mid_dirty']
            interest_rate_bump.setValue(0.0001)
            price_base_1bp_u = fixed_rate_bond.NPV()
            bond_convexity = (price_base_1bp_u + price_base_1bp - 2 * price_base) * 1000000 / row['mid_dirty'] *100
            sorted_details_frame.loc[index, 'scen_dv01'] = dv01
            sorted_details_frame.loc[index, 'scen_duration'] = bond_duration
            sorted_details_frame.loc[index, 'scen_convexity'] = bond_convexity
        else:
            print("Please enter a valid model")

    return sorted_details_frame

def calc_yield_to_worst(
            details: dict,
            pc_schedule: pd.DataFrame,
            bond_clean_price: float,
            calc_date: ql.Date):
    '''Computes yield-to-worst and workout date for fixed rate callable bonds.
    '''    
    
    maturity_date =  details['maturity']
    yield_to_maturity = details['yield_to_maturity']

    workout_date = maturity_date
    yield_to_worst = yield_to_maturity        
    
    # keep schedules for used bond only
    used_pc_schedule = pc_schedule[pc_schedule['figi'] == details['figi']]
    
    for _, row in used_pc_schedule.iterrows():
        call_date = get_ql_date(row['call_date'])
        if call_date > calc_date:
            
            # Create a call scenario details df
            call_scenario_details = details.copy()
            call_scenario_details['maturity'] = row['call_date']
            call_scenario_bond = create_bond_from_symbology(call_scenario_details)
            
            # scenario_yield
            call_scenario_yield = call_scenario_bond.bondYield(bond_clean_price, call_scenario_bond.dayCounter(), ql.Compounded, ql.Semiannual) * 100
                                            
            # Update yield_to_worst and workout_date if needed
            if call_scenario_yield < yield_to_worst:
                print('Found new workout date:', details['figi'], workout_date, call_date.to_date(), yield_to_worst, call_scenario_yield)
                
                yield_to_worst = call_scenario_yield
                workout_date = call_date.to_date()                                                            
                
    return workout_date, yield_to_worst


def get_yield_curve_details_df(yield_curve, curve_dates=None):
    
    if(curve_dates == None):
        curve_dates = yield_curve.dates()

    dates = [d.to_date() for d in curve_dates]
    discounts = [round(yield_curve.discount(d), 3) for d in curve_dates]
    yearfracs = [round(yield_curve.timeFromReference(d), 3) for d in curve_dates]
    zeroRates = [round(yield_curve.zeroRate(d, yield_curve.dayCounter(), ql.Compounded).rate() * 100, 3) for d in curve_dates]

    yield_curve_details_df = pd.DataFrame(data={'Date': dates,
                             'YearFrac': yearfracs,
                             'DiscountFactor': discounts,
                             'ZeroRate': zeroRates})                             
    return yield_curve_details_df


def get_hazard_rates_df(hazard_rate_curve,calc_date):
    '''Return dataframe with calibrated hazard rates and survival probabilities'''
    
    CDS_day_count = ql.Actual360()
    
    hazard_list = [(hr[0].to_date(), 
                CDS_day_count.yearFraction(calc_date, hr[0]),
                hr[1] * 1e4,
                hazard_rate_curve.survivalProbability(hr[0])) for hr in hazard_rate_curve.nodes()]

    grid_dates, year_frac, hazard_rates, surv_probs = zip(*hazard_list)

    hazard_rates_df = pd.DataFrame(data={'Date': grid_dates, 
                                     'YearFrac': year_frac,
                                     'HazardRateBps': hazard_rates,                                     
                                     'SurvivalProb': surv_probs})
    return(hazard_rates_df)


def create_cds_obj(cds_start_date, cds_maturity_date):
    # Common CDS specs
    side = ql.Protection.Seller
    cds_recovery_rate = 0.4
    cds_face_notional = 100
    contractual_spread_bps = 100

    # Create common CDS schedule
    cds_schedule = ql.MakeSchedule(cds_start_date, cds_maturity_date, ql.Period('3M'),
                                ql.Quarterly, ql.TARGET(), ql.Following, ql.Unadjusted, ql.DateGeneration.TwentiethIMM)

    # Create common CDS object
    cds_obj = ql.CreditDefaultSwap(side, cds_face_notional, contractual_spread_bps / 1e4, cds_schedule, ql.Following, ql.Actual360())
    return cds_obj


def create_cds_pricing_engine(calc_date, sofr_yield_curve_handle, cds_par_spread_5y_bps, cds_recovery_rate = 0.4, cds_tenor = 5):
    '''Calibrate hazard rate curve and create cds pricing engines from CDS Par Spreads'''

    CDS_tenor = ql.Period(cds_tenor, ql.Years)

    CDS_helpers = [ql.SpreadCdsHelper(cds_par_spread_5y_bps * 1e-4, CDS_tenor, 2, ql.TARGET(),
                                  ql.Quarterly, ql.Following, ql.DateGeneration.TwentiethIMM, ql.Actual360(), cds_recovery_rate, sofr_yield_curve_handle)]

    # bootstrap hazard_rate_curve
    hazard_rate_curve = ql.PiecewiseFlatHazardRate(calc_date, CDS_helpers, ql.Actual360())
    hazard_rate_curve.enableExtrapolation()

    # Create CDS pricing engine
    default_prob_curve_handle = ql.DefaultProbabilityTermStructureHandle(hazard_rate_curve)
    cds_pricing_engine = ql.MidPointCdsEngine(default_prob_curve_handle, cds_recovery_rate, sofr_yield_curve_handle)

    return(cds_pricing_engine)



#Create a function to price each OTR treasruy using the above bond engine. Extend the datafram with a column 'calc_mid' with the calculated mid price
def price_bond(symbology_df: pd.DataFrame, bond_engine, calc_date: ql.Date, risky_bond=False):
    
    ql.Settings.instance().evaluationDate = calc_date
    
    # Sort dataframe by maturity
    sorted_details_frame = symbology_df.copy() 
    
    for index, row in sorted_details_frame.iterrows():
        fixed_rate_bond = create_bond_from_symbology(row)
        fixed_rate_bond.setPricingEngine(bond_engine)
        pv_engine = fixed_rate_bond.NPV()
        
        sorted_details_frame.loc[index, 'calc_mid'] = pv_engine
            
            
    return sorted_details_frame


def calc_scenario_sensi(calc_date,sym_md_df, tsy_yield_curve_handle, default_prob_curve_handle,cds_par_spd,sofr_yield_curve_handle, flat_recovery_rate=0.4,flat_rec_bump=0):
    def bump_down_engine(bump=-0.0001):
        # Bump interest rate by -1bps (parallel shift)
        interest_rate_scenario_1bp_down = ql.SimpleQuote(bump)
        tsy_yield_curve_handle_1bp_down = ql.YieldTermStructureHandle(ql.ZeroSpreadedTermStructure(tsy_yield_curve_handle, ql.QuoteHandle(interest_rate_scenario_1bp_down)))
        risky_bond_engine_1bp_down = ql.RiskyBondEngine(default_prob_curve_handle, flat_recovery_rate, tsy_yield_curve_handle_1bp_down)
        return risky_bond_engine_1bp_down
        
    def bump_up_engine(bump=0.0001):
        # Bump interest rate by +1bps (parallel shift)
        interest_rate_scenario_1bp_up = ql.SimpleQuote(bump)
        tsy_yield_curve_handle_1bp_up = ql.YieldTermStructureHandle(ql.ZeroSpreadedTermStructure(tsy_yield_curve_handle, ql.QuoteHandle(interest_rate_scenario_1bp_up)))
        risky_bond_engine_1bp_up = ql.RiskyBondEngine(default_prob_curve_handle, flat_recovery_rate, tsy_yield_curve_handle_1bp_up)
        return risky_bond_engine_1bp_up
    
    def flat_engine(flat_rec_bump=0):
        # Flat interest rate
        interest_rate_scenario_flat = ql.SimpleQuote(0.0)
        tsy_yield_curve_handle_flat = ql.YieldTermStructureHandle(ql.ZeroSpreadedTermStructure(tsy_yield_curve_handle, ql.QuoteHandle(interest_rate_scenario_flat)))
        risky_bond_engine_flat = ql.RiskyBondEngine(default_prob_curve_handle, flat_recovery_rate+flat_rec_bump, tsy_yield_curve_handle_flat)
        return risky_bond_engine_flat
    
    def cds_par_bump_down(bump=-1):
        cds_par_spreads_1bp_down = [ps - 1 for ps in cds_par_spd]
        hazard_rate_curve_1bp_down = calibrate_cds_hazard_rate_curve(calc_date, sofr_yield_curve_handle, cds_par_spreads_1bp_down, flat_recovery_rate)
        default_prob_curve_handle_1bp_down = ql.DefaultProbabilityTermStructureHandle(hazard_rate_curve_1bp_down)
        risky_bond_engine_cds_1bp_down = ql.RiskyBondEngine(default_prob_curve_handle_1bp_down, flat_recovery_rate, tsy_yield_curve_handle)
        return risky_bond_engine_cds_1bp_down
        
    
    model_prices_1bp_up = []
    model_prices_1bp_down = []
    model_ir01 = []
    model_duration = []
    model_convexity = []
    model_analytic_duration = []
    model_analytic_convexity = []
    model_hr01 = []
    model_cs01 = []
    model_rec01 = []
    
    bonds = [create_bond_from_symbology(df_row.to_dict()) for index, df_row in sym_md_df.iterrows()]
    
    for i in range(0, len(bonds)):
        fixed_rate_bond = bonds[i]
    
        # Calc model dirty price for base case
        fixed_rate_bond.setPricingEngine(flat_engine())
        dirty_price_base = fixed_rate_bond.dirtyPrice()
        corpBondModelPrice = round(fixed_rate_bond.cleanPrice(), 3)
        corpBondModelYield = round(fixed_rate_bond.bondYield(corpBondModelPrice, ql.Thirty360(ql.Thirty360.USA), ql.Compounded, ql.Semiannual) * 100, 3)

        # Compute analytical duration and convexity (optional metrics)
        bond_yield_rate = ql.InterestRate(corpBondModelYield/100, ql.ActualActual(ql.ActualActual.ISMA), ql.Compounded, ql.Semiannual)
        analytic_duration = ql.BondFunctions.duration(fixed_rate_bond, bond_yield_rate)
        analytic_convexity = ql.BondFunctions.convexity(fixed_rate_bond, bond_yield_rate)

        # Scenario: 1bp down
        fixed_rate_bond.setPricingEngine(bump_down_engine())   
        price_1bp_down = fixed_rate_bond.cleanPrice()
        model_prices_1bp_down.append(price_1bp_down)
        
        # Scenario: 10bp down
        fixed_rate_bond.setPricingEngine(bump_down_engine(-0.001))
        price_10bp_down = fixed_rate_bond.cleanPrice()
        
        # Scenario: 1bp up
        fixed_rate_bond.setPricingEngine(bump_up_engine())
        price_1bp_up = fixed_rate_bond.cleanPrice()
        model_prices_1bp_up.append(price_1bp_up)
        
        # Scenario: 10bp up
        fixed_rate_bond.setPricingEngine(bump_up_engine(0.001))
        price_10bp_up = fixed_rate_bond.cleanPrice()
        
        # Scenario: CDS 1bp down
        fixed_rate_bond.setPricingEngine(cds_par_bump_down())
        price_cds_1bp_down = fixed_rate_bond.cleanPrice()
        yield_cds_1bp_down = fixed_rate_bond.bondYield(price_cds_1bp_down, ql.Thirty360(ql.Thirty360.USA), ql.Compounded, ql.Semiannual) * 100
        price_diff_cds_1bp_down = price_cds_1bp_down - corpBondModelPrice
        yield_diff_cds_1bp_down = yield_cds_1bp_down - corpBondModelYield
        
        # Scenario: Recovery rate 1% up
        fixed_rate_bond.setPricingEngine(flat_engine(0.01))
        price_rec_1p_up = fixed_rate_bond.cleanPrice()
        
        # Compute scenario delta/gamma sensitivities
        price_base = corpBondModelPrice
        ir01 = (price_1bp_down - price_base) * 1e4 / 100
        duration = ir01 / dirty_price_base * 100
        # Convexity
        gamma_1bp = (price_10bp_down - 2*price_base + price_10bp_up) * 1e6 / 100
        convexity = gamma_1bp / dirty_price_base * 100
        # HR01/CS01
        hr01 = - price_diff_cds_1bp_down / yield_diff_cds_1bp_down
        cs01 = price_diff_cds_1bp_down * 1e4 / 100
        # Rec01
        rec01 = (price_rec_1p_up - price_base)
        
        model_ir01.append(ir01)
        model_duration.append(duration)
        model_convexity.append(convexity)    
        model_analytic_duration.append(analytic_duration)
        model_analytic_convexity.append(analytic_convexity)
        model_hr01.append(hr01)
        model_cs01.append(cs01)
        model_rec01.append(rec01)
    
    sym_md_df['IRO1'] = model_ir01
    sym_md_df['Duration'] = model_duration
    sym_md_df['Convexity'] = model_convexity
    sym_md_df['Analytic Duration'] = model_analytic_duration
    sym_md_df['Analytic Convexity'] = model_analytic_convexity
    sym_md_df['HR01'] = model_hr01
    sym_md_df['CS01'] = model_cs01
    sym_md_df['Rec01'] = model_rec01
    
    return sym_md_df


def nelson_siegel(params, maturity):
    ''' params = (theta1, theta2, theta3, lambda)'''        
    if(maturity == 0 or params[3] <= 0):
        slope_1 = 1
        curvature = 0
    else:
        slope_1 = (1 - np.exp(-maturity/params[3]))/(maturity/params[3])
        curvature = slope_1 - np.exp(-maturity/params[3])

    total_value = params[0] + params[1] * slope_1 + params[2] * curvature
    
    return total_value

def create_nelson_siegel_curve(calc_date, nelson_siegel_params):
    ''' nelson_siegel_params = (theta1, theta2, theta3, lambda)'''            
    nelson_siegel_surv_prob_dates = [calc_date + ql.Period(T , ql.Years) for T in range(31)]
    nelson_siegel_average_hazard_rates = [nelson_siegel(nelson_siegel_params, T) for T in range(31)]
    nelson_siegel_surv_prob_levels = [np.exp(-T * nelson_siegel_average_hazard_rates[T]) for T in range(31)]
    
    # cap and floor survival probs
    nelson_siegel_surv_prob_levels = [max(min(x,1),1e-8) for x in nelson_siegel_surv_prob_levels]

    # nelson_siegel_surv_prob_curve
    nelson_siegel_credit_curve = ql.SurvivalProbabilityCurve(nelson_siegel_surv_prob_dates, nelson_siegel_surv_prob_levels, ql.Actual360(), ql.TARGET())
    nelson_siegel_credit_curve.enableExtrapolation()
    nelson_siegel_credit_curve_handle = ql.DefaultProbabilityTermStructureHandle(nelson_siegel_credit_curve)
    
    return(nelson_siegel_credit_curve_handle)


def calculate_nelson_siegel_model_prices_and_yields(nelson_siegel_params, 
                      calc_date, 
                      fixed_rate_bond_objects, 
                      tsy_yield_curve_handle, 
                      bond_recovery_rate = 0.4):
    
    # nelson_siegel_surv_prob_curve_handle
    nelson_siegel_surv_prob_curve_handle = create_nelson_siegel_curve(calc_date, nelson_siegel_params)
    
    # nelson_siegel_risky_bond_engine
    nelson_siegel_risky_bond_engine = ql.RiskyBondEngine(nelson_siegel_surv_prob_curve_handle, bond_recovery_rate, tsy_yield_curve_handle)
    
    bond_model_prices = []
    bond_model_yields = []
    
    for fixed_rate_bond in fixed_rate_bond_objects:
        fixed_rate_bond.setPricingEngine(nelson_siegel_risky_bond_engine)
        
        bond_price = fixed_rate_bond.cleanPrice()                
        bond_yield = fixed_rate_bond.bondYield(bond_price, ql.Thirty360(ql.Thirty360.USA), ql.Compounded, ql.Semiannual) * 100
        
        bond_model_prices.append(bond_price)
        bond_model_yields.append(bond_yield)
    
    return(bond_model_prices, bond_model_yields)

def nelson_siegel_sse(nelson_siegel_params, 
                      calc_date, 
                      fixed_rate_bond_objects, 
                      market_prices, 
                      calib_weights,
                      tsy_yield_curve_handle, 
                      bond_recovery_rate = 0.4):
    
    # bond_model_prices
    bond_model_prices, bond_model_yields = calculate_nelson_siegel_model_prices_and_yields(nelson_siegel_params, 
                      calc_date, 
                      fixed_rate_bond_objects, 
                      tsy_yield_curve_handle, 
                      bond_recovery_rate)
    # sse    
    sse = 0
    
    for i in range(len(market_prices)):
        model_error = market_prices[i] - bond_model_prices[i]                
        sse += model_error * model_error * calib_weights[i]                        
    
    return(sse)    


def create_bonds_and_weights(bond_details, tsy_yield_curve_handle):
    
    # risk_free_bond_engine
    risk_free_bond_engine = ql.DiscountingBondEngine(tsy_yield_curve_handle)


    fixed_rate_bond_objects = []
    bond_market_prices = []    
    bond_yields = []
    bond_DV01s = []    
    bond_durations = []    
    
    for index,row in bond_details.iterrows():
        fixed_rate_bond = create_bond_from_symbology(row)
        fixed_rate_bond.setPricingEngine(risk_free_bond_engine)
        
        fixed_rate_bond_objects.append(fixed_rate_bond)
        
        bond_price = row['market_price']                
        bond_yield = fixed_rate_bond.bondYield(bond_price, ql.Thirty360(ql.Thirty360.USA), ql.Compounded, ql.Semiannual) * 100
        bond_yield_rate = ql.InterestRate(bond_yield/100, ql.ActualActual(ql.ActualActual.ISMA), ql.Compounded, ql.Semiannual)
        bond_duration = ql.BondFunctions.duration(fixed_rate_bond, bond_yield_rate)
        bond_DV01   = fixed_rate_bond.dirtyPrice() * bond_duration
        
        bond_market_prices.append(bond_price)
        bond_yields.append(bond_yield)
        bond_DV01s.append(bond_DV01)
        bond_durations.append(bond_duration)   
        
    # calib_weights: down-weight durations < 2 years, since the calibrated US treasury does not have quotes before 2Y
    calib_weights = [1 / max(d, 2) for d in bond_durations]
    sum_calib_weights = sum(calib_weights)
    calib_weights = [x / sum_calib_weights for x in calib_weights]
    
    return(fixed_rate_bond_objects, calib_weights, bond_market_prices, bond_yields, bond_DV01s, bond_durations)
    
def calibrate_nelson_siegel_model(initial_nelson_siegel_params,
                                  calc_date, 
                                  bond_details, 
                                  tsy_yield_curve_handle, 
                                  bond_recovery_rate = 0.4):
    # create_bonds_and_weights
    fixed_rate_bond_objects, calib_weights, bond_market_prices, bond_yields, bond_DV01s, bond_durations = create_bonds_and_weights(bond_details, tsy_yield_curve_handle)
    
    # start calibration
    param_bounds = [(1e-3, 0.1), (-0.1, 0.1), (-0.1, 0.1), (1e-3, 10)]
            
    calib_results = minimize(nelson_siegel_sse,
                                            initial_nelson_siegel_params, 
                                            args = (calc_date, 
                                                    fixed_rate_bond_objects, 
                                                    bond_market_prices, 
                                                    calib_weights,
                                                    tsy_yield_curve_handle, 
                                                    bond_recovery_rate),
                                            bounds = param_bounds)


    return(calib_results)

def calc_d1_d2(A,r,sigma_A,T,L):
    d1 = (-np.log(L/A) + (r + 0.5 * sigma_A**2)* T ) / (sigma_A * np.sqrt(T))
    d2 = (-np.log(L/A) + (r - 0.5 * sigma_A**2)* T ) / (sigma_A * np.sqrt(T))    
    return (d1, d2)

def fairValueEquity(A,r,sigma_A,T,L):
    d1, d2 = calc_d1_d2(A,r,sigma_A,T,L)
    E0  = A * norm.cdf(d1) - np.exp(-r * T) * L * norm.cdf(d2)
    return E0

def fairValueRiskyBond(A,r,sigma_A,T,L):
    d1, d2 = calc_d1_d2(A,r,sigma_A,T,L)
    B0  = A * norm.cdf(-d1) + L * np.exp(-r * T) * norm.cdf(d2)
    
    return B0

def defaultProbability(A,r,sigma_A,T,K):
    d1, d2 = calc_d1_d2(A,r,sigma_A,T,K)
    default_prob = norm.cdf(-d2)
    
    return default_prob

def survivalProbability(A,r,sigma_A,T,L):
    return(1 - defaultProbability(A,r,sigma_A,T,L))

def distanceToDefault(A,r,sigma_A,T,K):
    d1, d2 = calc_d1_d2(A,r,sigma_A,T,K)        
    return(d2)

def riskyBondYield(A,r,sigma_A,T,K):
    B0 = fairValueRiskyBond(A,r,sigma_A,T,K)
    bond_yield = - np.log(B0/K) / T       
    return bond_yield

def riskyBondCreditSpread(A,r,sigma_A,T,K):
    bond_yield = riskyBondYield(A,r,sigma_A,T,K)    
    bond_credit_spread = bond_yield - r
    return bond_credit_spread

def flatHazardRate(A,r,sigma_A,T,K):
    survival_prob = survivalProbability(A,r,sigma_A,T,K)
    flat_hazard_rate = - np.log(survival_prob) / T
    return flat_hazard_rate

def expectedRecoveryRate(A,r,sigma_A,T,K):
    d1, d2 = calc_d1_d2(A,r,sigma_A,T,K)    
    exp_rec_rate = A / K * norm.cdf(-d1)/norm.cdf(-d2)
    return exp_rec_rate

def equityVolatility(A,r,sigma_A,T,K):
    d1, d2 = calc_d1_d2(A,r,sigma_A,T,K)    
    E0 = fairValueEquity(A,r,sigma_A,T,K)    
    sigma_E = (A / E0) * norm.cdf(d1) * sigma_A
    return sigma_E

def cds_prem_def_pv(calc_date, cds_obj, cdx_basket_dataframe, sofr_yield_curve_handle, cds_recovery_rate=0.4, cds_tenor=5):
    cds_premium_leg_pvs = []
    cds_default_leg_pvs = []
    cds_pvs = []
    
    #weighted pvs
    cdx_ig_5y_premium_leg_pv = 0
    cdx_ig_5y_default_leg_pv = 0
    cdx_ig_5y_pv = 0
    
    for i in range(cdx_basket_dataframe.shape[0]):
        par_spread = cdx_basket_dataframe.iloc[i]['cds_par_spread_5y']
        cds_basket_weight = cdx_basket_dataframe.iloc[i]['index_weight'] /100
        # create engine
        cds_engine = create_cds_pricing_engine(calc_date, sofr_yield_curve_handle,par_spread, cds_recovery_rate, cds_tenor)
        # setPricingEngine
        cds_obj.setPricingEngine(cds_engine)
        
        # Calc individual CDS PVs
        cds_premium_leg_pv = cds_obj.couponLegNPV()
        cds_default_leg_pv = -cds_obj.defaultLegNPV()
        cds_pv = cds_obj.NPV()
        
        # Weighted PVs
        cdx_ig_5y_premium_leg_pv += (cds_premium_leg_pv * cds_basket_weight)
        cdx_ig_5y_default_leg_pv += (cds_default_leg_pv * cds_basket_weight)
        cdx_ig_5y_pv += (cds_pv * cds_basket_weight)
        
        cds_premium_leg_pvs.append(cds_premium_leg_pv)
        cds_default_leg_pvs.append(cds_default_leg_pv)
        cds_pvs.append(cds_pv)
        
    # Add results to cdx dataframe
    cdx_basket_dataframe['premium_leg_pv'] = cds_premium_leg_pvs
    cdx_basket_dataframe['default_leg_pv'] = cds_default_leg_pvs
    cdx_basket_dataframe['cds_pv'] = cds_pvs
        
    print('CDS Premium Leg PV:', round(cdx_ig_5y_premium_leg_pv,3))
    print('CDS Default Leg PV:', round(cdx_ig_5y_default_leg_pv,3))
    print('CDS PV:', round(cdx_ig_5y_pv,3))
    contractual_spread_bps = 100
    par_spread = contractual_spread_bps * cdx_ig_5y_default_leg_pv / cdx_ig_5y_premium_leg_pv
    print('Par Spread:', round(par_spread,3))
    return cdx_basket_dataframe, cdx_ig_5y_premium_leg_pv, cdx_ig_5y_default_leg_pv, cdx_ig_5y_pv,par_spread


def get_nav(hyg_dataframe):
    # face notionals and weights
    bond_face_notionals = hyg_dataframe['face_notional']
    bond_face_notional_weights = hyg_dataframe['face_notional_weight']
    
    # bond objects and dirty prices
    hyg_bond_objects = []
    hyg_bond_dirty_prices = []
    
    # ETF intrinsic  NAV and Market Cap
    hyg_intrinsic_nav = 0
    hyg_intrinsic_market_cap = 0
    
    # bond objs
    
    for i, row in hyg_dataframe.iterrows():
        # create bond object
        bond_obj = create_bond_from_symbology(row)
        
        # bond_dirty_price                
        bond_yield = row['yield_to_maturity'] / 100
        bond_dirty_price = bond_obj.dirtyPrice(bond_yield, ql.Thirty360(ql.Thirty360.USA), ql.Compounded, ql.Semiannual)
        
        # hyg_basket_nav    
        hyg_intrinsic_nav += bond_dirty_price * bond_face_notional_weights[i] / 100
        
        # hyg_basket_market_cap    
        hyg_intrinsic_market_cap += bond_dirty_price * bond_face_notionals[i] / 100
        
        # Populate lists
        hyg_bond_objects.append(bond_obj)
        hyg_bond_dirty_prices.append(bond_dirty_price)
        
    ## Add dirty prices to hyg_df
    hyg_dataframe['dirty_price'] = hyg_bond_dirty_prices
    hyg_dataframe['bond_obj'] = hyg_bond_objects
    
    return hyg_dataframe, hyg_intrinsic_nav, hyg_intrinsic_market_cap


def calc_etf_nav_from_yield(etf_yield, hyg_bond_objects, bond_face_notional_weights):
    
    # etf_intrinsic_nav
    etf_intrinsic_nav = 0
    
    # loop over bonds
    for i in range(len(hyg_bond_objects)):
        bond_object = hyg_bond_objects[i]
        # calc bond_dirty_price
        bond_dirty_price = bond_object.dirtyPrice(etf_yield, ql.Thirty360(ql.Thirty360.USA), ql.Compounded, ql.Semiannual)        
        
        # update etf_intrinsic_nav
        etf_intrinsic_nav += bond_dirty_price * bond_face_notional_weights[i] / 100
        
    return(etf_intrinsic_nav)


def calc_etf_nav_from_zspread(etf_zspread, hyg_bond_objects, bond_face_notional_weights, tsy_yield_curve_handle):
    
    # Add z-spread to tsy_yield_curve_handle and obtain yield_curve_bumped_handle
    zspread_quote_handle = ql.QuoteHandle(ql.SimpleQuote(etf_zspread))
    yield_curve_bumped = ql.ZeroSpreadedTermStructure(tsy_yield_curve_handle, zspread_quote_handle, ql.Compounded, ql.Semiannual)
    yield_curve_bumped_handle = ql.YieldTermStructureHandle(yield_curve_bumped)
    
    # zspread_bond_engine
    zspread_bond_engine = ql.DiscountingBondEngine(yield_curve_bumped_handle)
    
    # etf_intrinsic_nav
    etf_intrinsic_nav = 0
    
    # loop over bonds
    for i in range(len(hyg_bond_objects)):        
        
        bond_object = hyg_bond_objects[i]
        bond_object.setPricingEngine(zspread_bond_engine)
        
        # calc bond_dirty_price        
        bond_dirty_price = bond_object.dirtyPrice()                
        
        # update etf_intrinsic_nav
        etf_intrinsic_nav += bond_dirty_price * bond_face_notional_weights[i] / 100
        
    return(etf_intrinsic_nav)
    