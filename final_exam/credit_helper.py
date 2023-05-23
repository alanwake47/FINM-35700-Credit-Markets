
import QuantLib as ql
import numpy as np
import pandas as pd
import datetime as dt

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


def create_cds_object(cds_start_date, cds_maturity_date):
    # Common CDS specs
    side = ql.Protection.Seller
    cds_recovery_rate = 0.4
    cds_face_notional = 100
    contractual_spread_bps = 100

    # Create common CDS schedule
    cds_schedule = ql.MakeSchedule(cds_start_date, cds_maturity_date, ql.Period('3M'),
                                ql.Quarterly, ql.TARGET(), ql.Following, ql.Unadjusted, ql.DateGeneration.TwentiethIMM)

    # Create common CDS object
    cds_object = ql.CreditDefaultSwap(side, cds_face_notional, contractual_spread_bps / 1e4, cds_schedule, ql.Following, ql.Actual360())
    return cds_object


def create_cds_pricing_engine(calc_date, sofr_yield_curve_handle, cds_par_spread_5y_bps, cds_recovery_rate = 0.4):
    '''Calibrate hazard rate curve and create cds pricing engines from 5Y CDS Par Spreads'''

    # CDS tenor: 5Y only
    CDS_tenor_5Y = ql.Period(5, ql.Years)

    CDS_helpers = [ql.SpreadCdsHelper(cds_par_spread_5y_bps * 1e-4, CDS_tenor_5Y, 2, ql.TARGET(),
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
