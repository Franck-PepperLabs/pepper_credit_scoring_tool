# client_viewer.py
from _dashboard_commons import *
from _api_getters import get_table_names, get_client_data
from _user_controls import client_selector
from _displays import pivoted_display


def client_viewer_main():
    log_main_run(this_f_name())

    # Create a title for the app
    st.title("Client Data Viewer")

    # Filter the clients list
    clients = client_selector()

    # Client Selection
    client_id = st.selectbox(
        f"Select a Client ({clients.shape[0]})", clients.index)
    
    # Display the client data from the 7 tables
    for table_name in get_table_names():
        st.header(f"`{table_name.upper()}` data")
        client_data = get_client_data(table_name, client_id)
        st.dataframe(client_data)

    # Bureau balance pivoted display
    pivoted_display(
        client_id, "bureau_balance",
        "SK_ID_BUREAU", "MONTHS_BALANCE", "STATUS"
    )

    # POS CASH balance pivoted display
    pivoted_display(
        client_id, "pos_cash_balance",
        "SK_ID_PREV", "MONTHS_BALANCE", "NAME_CONTRACT_STATUS"
    )

    # Credit Car balance pivoted display
    pivoted_display(
        client_id, "credit_card_balance",
        "SK_ID_PREV", "MONTHS_BALANCE", "NAME_CONTRACT_STATUS"
    )

    # Installments payments pivoted display
    # table_name = "installments_payments"
    # st.header(f"`{table_name.upper()}` pivoted data")
    # client_data = get_client_data(table_name, client_id)
    # client_data.MONTHS_BALANCE = -client_data.MONTHS_BALANCE
    # client_data = client_data.pivot(
    #     index="SK_ID_PREV", columns="MONTHS_BALANCE", values="STATUS")
    # st.dataframe(client_data)


if __name__ == "__main__":
    init_session()
    client_viewer_main()
