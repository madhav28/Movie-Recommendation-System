import streamlit as st
import pandas as pd

if 'user_id' not in st.session_state:
    st.session_state.user_id = None


def login(username, password):
    return username >= 1 and username <= 10 and password == "password"


def main():

    is_logged_in = st.session_state.get("is_logged_in", False)

    if not is_logged_in:
        st.title("Movie Recommendation System")

        username = st.text_input("User ID:", value="5")
        password = st.text_input(
            "Password:", type="password", value="password")

        login_button_clicked = st.button("Login")

        if login_button_clicked:
            st.session_state.user_id = int(username)
            if login(int(username), password):
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
    user_id = st.session_state.user_id

    st.title("Movie Recommendation System")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(['Introduction', 'Data Visualisation',
                                            'Algorithms', 'Results and Discussion',
                                            'Conclusion and Future Work'])

    with tab1:
        st.markdown("#### Problem Definition:")
        st.markdown("Our project focuses on the development of a sophisticated movie recommendation system,\
                    aimed at enhancing user experience and engagement within the context of contemporary\
                    businesses. Leveraging the power of recommendation algorithms, we aim to provide a\
                    comprehensive and user-friendly platform that adapts to evolving customer preferences. The\
                    primary objectives are to optimize user satisfaction, boost user retention rates, and align with\
                    the dynamic tastes of our customer base")
        st.markdown("#### Data Source:")
        st.markdown(
            "Movie Lens 25M Dataset - https://grouplens.org/datasets/movielens/25m/")

    with tab2:
        st.markdown("#### Distributions for Movies:")
        st.image("movie_release.png")
        st.image("movie_genres.png")
        st.markdown("#### Distributions for Movie Rating:")
        st.image("average_rating.png")
        st.image("movie_rating_rel.png")
        st.image("movie_rating_genre.png")
        st.image("most_rated_movies.png")

    with tab3:
        tab_a1, tab_a2, tab_a3 = st.tabs(["Collaborative Filtering",
                                          "Content-Based Filtering", "Hybrid"])
        with tab_a1:
            st.markdown("""
                        Collaborative filtering is a technique used in recommendation systems to predict a user's preferences or interests by leveraging the preferences and behaviors of a group of users. The underlying idea is that users who have agreed in the past on certain issues tend to agree again in the future. This method assumes that if a user A has similar preferences to a user B on a certain issue, A is more likely to share B's preferences on a different issue as well.

            There are two main types of collaborative filtering: user-based and item-based.

            User-Based Collaborative Filtering:

            This approach recommends items to a target user based on the preferences and behavior of users who are similar to that target user.
            The system identifies users with similar preferences to the target user and recommends items that those similar users have liked or interacted with.
            The similarity between users is often calculated using different metrics such as cosine similarity or Pearson correlation.
            Item-Based Collaborative Filtering:

            In this approach, the system recommends items similar to those that a user has liked or interacted with in the past.
            It identifies items that are similar to the ones the target user has shown interest in and recommends them.
            Similarity between items is also calculated using metrics like cosine similarity or Pearson correlation.
                        """)

            tab_a1_mat, tab_a1_dl = st.tabs(
                ["Matrix Factorization", "Deep learning based"])
            with tab_a1_mat:
                st.markdown("#### About:")
                st.markdown("#### Top 10 Recommendations (for User ID = " +
                            str(user_id)+") are:  ")
                colab_matrix_fact = pd.read_csv("colab_matrix_fact.csv")
                colab_matrix_fact = colab_matrix_fact[colab_matrix_fact['users'] == user_id]
                recommendations = {}
                top_10 = []
                for column in colab_matrix_fact.columns:
                    if column == 'users':
                        continue
                    top_10.append(colab_matrix_fact[column].ravel()[0])

                recommendations = {"Movie Recommendation": top_10}
                recommendations = pd.DataFrame(recommendations)
                recommendations.set_index(pd.RangeIndex(
                    start=1, stop=11, step=1), inplace=True)
                recommendations = recommendations.rename_axis('#')

                st.dataframe(recommendations)

            with tab_a1_dl:
                st.markdown("#### About:")
                st.markdown("#### Top 10 Recommendations (for User ID = " +
                            str(user_id)+") are:  ")
                colab_dl = pd.read_csv("colab_dl.csv")
                colab_dl = colab_dl[colab_dl['user_id'] == user_id]
                recommendations = {}
                top_10 = []
                for column in colab_dl.columns:
                    if column == 'user_id':
                        continue
                    top_10.append(colab_dl[column].ravel()[0])

                recommendations = {"Movie Recommendation": top_10}
                recommendations = pd.DataFrame(recommendations)
                recommendations.set_index(pd.RangeIndex(
                    start=1, stop=11, step=1), inplace=True)
                recommendations = recommendations.rename_axis('#')

                st.dataframe(recommendations)

        with tab_a2:
            st.markdown("""
                        Content-based filtering is another approach used in recommendation systems, and it relies on the characteristics or features of items and users to make recommendations. Unlike collaborative filtering, content-based filtering doesn't require information about the preferences or behaviors of other users. Instead, it focuses on the properties of items and the explicit profile of the user.
                        """)
        with tab_a3:
            st.markdown("""
                        A hybrid recommendation system is an approach that combines multiple recommendation techniques to overcome the limitations of individual methods and provide more accurate and diverse recommendations. By leveraging the strengths of different recommendation algorithms, hybrid models aim to enhance overall performance and address challenges such as the cold start problem, sparsity of data, and the diversity of recommendations. 
                        """)


if __name__ == "__main__":
    main()
