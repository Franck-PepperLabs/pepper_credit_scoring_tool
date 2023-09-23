from _dashboard_commons import *
from client_default_risk import client_default_risk_main
from client_viewer import client_viewer_main
from table_viewer import table_viewer_main


def dashboard_main():
    # Sidebar navigation bar to choose the page
    viewers = {
        "Client Default Risk Viewer": client_default_risk_main,
        "Client Info Explorer": client_viewer_main,
        "Table Explorer": table_viewer_main,
        # "Feature Explorer": feature_viewer_main
    }
    
    # Select a page to view from the sidebar
    viewer_name = st.sidebar.selectbox("Current View", viewers)

    # Display the selected page
    viewers[viewer_name]()


if __name__ == "__main__":
    # Initialize the session and run the main dashboard
    init_session()
    dashboard_main()