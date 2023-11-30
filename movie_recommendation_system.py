import streamlit as st


def login(username, password):
    return username == "username" and password == "password"


def main():

    is_logged_in = st.session_state.get("is_logged_in", False)

    if not is_logged_in:
        st.title("Movie Recommendation System")

        username = st.text_input("Username:", value="username")
        password = st.text_input(
            "Password:", type="password", value="password")

        login_button_clicked = st.button("Login")

        if login_button_clicked:
            if login(username, password):
                st.success("Login successful!")
                st.session_state.is_logged_in = True
                st.experimental_rerun()
            else:
                st.error("Login failed. Please try again.")
    else:
        col1, col2 = st.columns([4, 1])

        if col2.button("Logout"):
            st.session_state.is_logged_in = False
            st.experimental_rerun()
        app_content()


def app_content():
    st.title("Movie Recommendation System")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(['Introduction', 'Data Visualisation',
                                            'Algorithm', 'Results and Discussion',
                                            'Conclusion and Future Work'])

    with tab1:
        st.markdown("#### Problem Definition:")
        st.markdown("Our project focuses on the development of a sophisticated movie recommendation system,\
                    aimed at enhancing user experience and engagement within the context of contemporary\
                    businesses. Leveraging the power of recommendation algorithms, we aim to provide a\
                    comprehensive and user-friendly platform that adapts to evolving customer preferences. The\
                    primary objectives are to optimize user satisfaction, boost user retention rates, and align with\
                    the dynamic tastes of our customer base")


if __name__ == "__main__":
    main()
