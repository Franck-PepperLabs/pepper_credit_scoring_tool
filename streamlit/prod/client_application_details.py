# client_viewer.py
from _dashboard_commons import *
from _api_getters import get_table_names, get_client_data
from _user_controls import client_selector





def flag_symbol(flag):
    return "✘" if flag else "✔"
    


def print_title(title: str) -> None:
    st.title(title)

def print_subtitle(subtitle: str) -> None:
    st.header(subtitle)

def display_key_val(key, val):
    # TODO : complete with other type cases
    val_str = f"{val:n}" if isinstance(val, int) else f"{val}"
    st.write(f"**{key}**: {val_str}")



def client_application_details_main():
    log_main_run(this_f_name())

    # Create a title for the app
    st.title("Client Application Details")

    # Filter the clients list
    clients = client_selector()

    # Client Selection
    client_id = st.selectbox(
        f"Select a Client ({clients.shape[0]})", clients.index)

    # Retrieve client's application data
    client_application_data = get_client_data("application", client_id)

    def value(var_name):
        val = client_application_data[var_name].values[0]
        if f"{val}" == "nan":
            return "_"
        return int(val) if isinstance(val, float) and val > 1000 else val

    def days_in_years(days_name):
        days = value(days_name)
        years = round(-days/365.243, 2)
        return f"{years} years"

    def labeled_flag(flag_name, label, reverse=False):
        flag = value(flag_name)
        if reverse:
            flag = not flag
        return f"{flag_symbol(flag)} {label}"

    #print_title("View applicant infos")
    
    print_subtitle(f"Applicant ID: {client_id}")
    
    print_subtitle("Person")
    display_key_val("Sex", "Male" if value("CODE_GENDER") == "M" else "Female")
    display_key_val("Age", days_in_years("DAYS_BIRTH"))
    display_key_val("Employed", days_in_years("DAYS_EMPLOYED"))
    display_key_val("Registered", days_in_years("DAYS_REGISTRATION"))
    display_key_val("Identity docs", days_in_years("DAYS_ID_PUBLISH"))

    st.markdown("- **Sex** : Male" if value("CODE_GENDER") == "M" else "- **Sex** : Female", unsafe_allow_html=True)
    st.markdown(f"- **Age** : {days_in_years('DAYS_BIRTH')}", unsafe_allow_html=True)
    st.markdown(f"- **Employed** : {days_in_years('DAYS_EMPLOYED')}", unsafe_allow_html=True)
    st.markdown(f"- **Registered** : {days_in_years('DAYS_REGISTRATION')}", unsafe_allow_html=True)
    st.markdown(f"- **Identity docs** : {days_in_years('DAYS_ID_PUBLISH')}", unsafe_allow_html=True)

    print_subtitle("Application")
    
    display_key_val("Contract type", value("NAME_CONTRACT_TYPE"))
    display_key_val("Co-contractor", value("NAME_TYPE_SUITE"))
    start_date = (
        f"{value('WEEKDAY_APPR_PROCESS_START').title()} "
        f"at {value('HOUR_APPR_PROCESS_START')}:00"
    )
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

    st.write(labeled_flag(
        "REG_CITY_NOT_LIVE_CITY",
        "Permanent and contact adresses are in the same city",
        reverse=True
    ))
    st.write(labeled_flag(
        "REG_CITY_NOT_WORK_CITY",
        "Permanent and work adresses are in the same city",
        reverse=True
    ))
    st.write(labeled_flag(
        "LIVE_CITY_NOT_WORK_CITY",
        "Contact and work adresses are in the same city",
        reverse=True
    ))
    
    st.write(labeled_flag(
        "REG_REGION_NOT_LIVE_REGION",
        "Permanent and contact adresses are in the same region",
        reverse=True
    ))
    st.write(labeled_flag(
        "REG_REGION_NOT_WORK_REGION",
        "Permanent and work adresses are in the same region",
        reverse=True
    ))
    st.write(labeled_flag(
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
    st.write(", ".join(labeled_doc_flags[:5]))
    st.write(", ".join(labeled_doc_flags[5:10]))
    st.write(", ".join(labeled_doc_flags[10:15]))
    st.write(", ".join(labeled_doc_flags[15:]))
