from home_credit.load import get_table
from pepper.utils import print_title, print_subtitle, display_key_val


def view_applicant_info(applicant_id: int):
    data = get_table("application")
    applicant_data = data[data.SK_ID_CURR == applicant_id]
    
    def value(var_name):
        val = applicant_data[var_name].values[0]
        if f"{val}" == "nan":
            return "_"
        return int(val) if isinstance(val, float) and val > 1000 else val
    
    def flag_symbol(flag):
        return "✘" if flag else "✔"
        
    def labeled_flag(flag_name, label, reverse=False):
        flag = value(flag_name)
        if reverse:
            flag = not flag
        return f"{flag_symbol(flag)} {label}"
    
    def days_in_years(days_name):
        days = value(days_name)
        years = round(-days/365.243, 2)
        return f"{years} years"
    
    print_title("View applicant infos")
    
    print_subtitle(f"Applicant ID: {applicant_id}")
    
    print_subtitle("Person")
    display_key_val("Sex", "Male" if value("CODE_GENDER") == "M" else "Female")
    display_key_val("Age", days_in_years("DAYS_BIRTH"))
    display_key_val("Employed", days_in_years("DAYS_EMPLOYED"))
    display_key_val("Registered", days_in_years("DAYS_REGISTRATION"))
    display_key_val("Identity docs", days_in_years("DAYS_ID_PUBLISH"))
    
    print_subtitle("Application")
    
    display_key_val("Contract type", value("NAME_CONTRACT_TYPE"))
    display_key_val("Co-contractor", value("NAME_TYPE_SUITE"))
    start_date = f"{value('WEEKDAY_APPR_PROCESS_START').title()} at {value('HOUR_APPR_PROCESS_START')}:00"
    display_key_val("Process start", start_date)

    print_subtitle("Financial statement")
    income = value("AMT_INCOME_TOTAL")
    loan = value("AMT_CREDIT")
    assets = value("AMT_GOODS_PRICE")
    annuity = value("AMT_ANNUITY")
    display_key_val("Annual income", income)
    display_key_val("Loan         ", loan)
    display_key_val("Assets       ", assets)
    display_key_val("Loan annuity ", annuity)
    display_key_val("Debt ratio   ", f"{round(100 * annuity / income)} %")
    display_key_val("Loan term    ", f"{round(loan / annuity, 1)} years")    

    print_subtitle("Contact")

    print(
        f"Phones provided  : "
        f"{labeled_flag('FLAG_MOBIL', 'Mobile')}, "
        f"{labeled_flag('FLAG_EMP_PHONE', 'Employer')}, "
        f"{labeled_flag('FLAG_WORK_PHONE', 'Work')}, "
        f"{labeled_flag('FLAG_PHONE', 'Home')}"
    )
    mobile_ok_flag = value("FLAG_CONT_MOBILE")
    print(f"Mobile reachable : {flag_symbol(mobile_ok_flag)}")
    email_flag = value("FLAG_EMAIL")
    print(f"Email provided   : {flag_symbol(email_flag)}")

    print_subtitle("Income, education and occupation")
    display_key_val("Income type", value("NAME_INCOME_TYPE"))
    display_key_val("Education type", value("NAME_EDUCATION_TYPE"))
    display_key_val("Occupation type", value("OCCUPATION_TYPE"))
    display_key_val("Organization type", value("ORGANIZATION_TYPE"))
    
    print_subtitle("Family")
    display_key_val("Family status", value("NAME_FAMILY_STATUS"))
    display_key_val("Number of family members", value("CNT_FAM_MEMBERS"))
    display_key_val("Number of children", value("CNT_CHILDREN"))
    
    print_subtitle("Housing")
    display_key_val("Housing type", value("NAME_HOUSING_TYPE"))
    
    print_subtitle("Region")
    # Numéro de région cf. mon analyse
    display_key_val("Population indice", value("REGION_POPULATION_RELATIVE"))
    display_key_val("Region rating", value("REGION_RATING_CLIENT"))
    display_key_val("City rating", value("REGION_RATING_CLIENT_W_CITY"))

    print_subtitle("Daily commute to work")

    print(labeled_flag(
        "REG_CITY_NOT_LIVE_CITY",
        "Permanent and contact adresses are in the same city",
        reverse=True
    ))
    print(labeled_flag(
        "REG_CITY_NOT_WORK_CITY",
        "Permanent and work adresses are in the same city",
        reverse=True
    ))
    print(labeled_flag(
        "LIVE_CITY_NOT_WORK_CITY",
        "Contact and work adresses are in the same city",
        reverse=True
    ))
    
    print(labeled_flag(
        "REG_REGION_NOT_LIVE_REGION",
        "Permanent and contact adresses are in the same region",
        reverse=True
    ))
    print(labeled_flag(
        "REG_REGION_NOT_WORK_REGION",
        "Permanent and work adresses are in the same region",
        reverse=True
    ))
    print(labeled_flag(
        "LIVE_REGION_NOT_WORK_REGION",
        "Contact and work adresses are in the same region",
        reverse=True
    ))

    print_subtitle("External ratings")
    display_key_val("External rating 1", value("EXT_SOURCE_1"))
    display_key_val("External rating 2", value("EXT_SOURCE_2"))
    display_key_val("External rating 3", value("EXT_SOURCE_3"))

    print_subtitle("Documents provided")
    labeled_doc_flags = [labeled_flag(f"FLAG_DOCUMENT_{i}", f"{i}") for i in range(2, 22)]
    print(", ".join(labeled_doc_flags[:5]))
    print(", ".join(labeled_doc_flags[5:10]))
    print(", ".join(labeled_doc_flags[10:15]))
    print(", ".join(labeled_doc_flags[15:]))