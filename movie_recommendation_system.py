import streamlit as st


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
            user_id = int(username)
            if login(user_id, password):
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
            with tab_a1_dl:
                st.markdown("#### About:")

        with tab_a2:
            st.markdown("""
                        Content-based filtering is another approach used in recommendation systems, and it relies on the characteristics or features of items and users to make recommendations. Unlike collaborative filtering, content-based filtering doesn't require information about the preferences or behaviors of other users. Instead, it focuses on the properties of items and the explicit profile of the user.
                        """)
        with tab_a3:
            st.markdown("""
                        A hybrid recommendation system is an approach that combines multiple recommendation techniques to overcome the limitations of individual methods and provide more accurate and diverse recommendations. By leveraging the strengths of different recommendation algorithms, hybrid models aim to enhance overall performance and address challenges such as the cold start problem, sparsity of data, and the diversity of recommendations. 
                        """)
            
        with tab4:
            st.markdown("""
                        
                        ### Collaborative Filtering:
                        The performance of the movie prediction models was rigorously evaluated using RMSE (Root Mean Square Error) as the benchmark metric, comparing true ratings against predicted ratings. The following is a summary of the results:
                        """)
            st.image("rmse_scores.png")


            st.markdown("""
                        #### Comparative Analysis:
                        In comparing these models, a few trends and patterns surfaced. Traditional collaborative filtering methods, such as user-based and item-based, struggled with scalability and sparsity issues, limiting their effectiveness. Matrix factorization provided a more nuanced approach by capturing latent factors, yet it fell short in addressing the complexities of diverse user preferences.

                        Deep learning models, on the other hand, exhibited superior performance. The intricate architectures allowed them to grasp subtle relationships, resulting in more accurate predictions. However, the computational demands and potential overfitting challenges should be carefully considered.
                        
                        #### Possible Problems:
                        Despite the advancements, certain challenges persist:

                        - **Cold Start Problem:** All collaborative filtering methods may struggle with new or rarely-rated items and users.
                        - **Scalability:** User-based collaborative filtering faces scalability issues with a growing user base.
                        - **Data Sparsity:** Matrix factorization and collaborative filtering models may encounter challenges in sparse datasets where user-item interactions are limited.
                        - **Interpretability:** Deep learning models, while powerful, often lack interpretability, making it challenging to explain recommendations to users.
                        """)
            
            st.markdown("""
                        ### Content-Based Filtering
                        Leveraging Natural Language Processing (NLP) for content-based filtering added an extra layer of sophistication to our movie recommendation system.

                        **Effectiveness of NLP Features:**\
                        The NLP-based content filtering method proved effective in extracting semantic information from textual data. This allowed the model to capture subtle nuances in movie content and recommend films that align more closely with users' preferences.
                        
                        **Consideration of Genres:**\
                        By incorporating genres and themes extracted through NLP, the model exhibited a nuanced understanding of user preferences beyond numerical ratings. This is particularly beneficial when users have diverse tastes that extend beyond traditional collaborative filtering signals.
                        
                        **User-Generated Tags:**\
                        The inclusion of user-generated tags further enriched the content-based approach, allowing the model to incorporate real-time, user-contributed descriptors. This dynamic aspect adds a layer of relevance that static metadata may lack.
                        
                        #### Comparative Analysis:
                        In comparison to collaborative filtering methods, the NLP-based content filtering method showcased competitive performance. It proved particularly effective in scenarios where traditional collaborative filtering faced challenges, such as sparse datasets and the cold start problem.

                        #### Possible Problems:
                        Despite its strengths, the content-based method using NLP is not without potential challenges:

                        - **Limited Coverage:** If movie descriptions are sparse or lack informative content, the model might struggle to generate meaningful recommendations.
                        - **Dependency on Textual Data Quality:** The quality of NLP features heavily depends on the richness and accuracy of textual data. Inaccuracies or biases in the data could impact recommendation quality.

                        """)

        with tab5:
            st.markdown("""
                        ## Conclusion:
                        In conclusion, the project has successfully developed and implemented a robust movie prediction system, leveraging collaborative filtering and Content-Based filtering techniques. The evaluation results showcase the effectiveness of deep learning models, closely followed by matrix factorization, in accurately predicting user ratings for movies. The algorithm has demonstrated the ability to provide recommendations that align well with human suggestions.

                        The incorporation of features such as similar movies, genres, and popularity has enhanced the algorithm's recommendation capabilities. The decision to train the models on the entire dataset has proven fruitful, contributing to improved accuracy on the test set.

                        The project marks a significant step forward in personalized movie recommendations, creating a user-friendly application that harnesses the power of collaborative filtering to enhance the movie-watching experience.

                        ## Future Works:
                        While the current models have shown promising results, there are several avenues for future improvements and enhancements:

                        **Real-Time Recommendations:**\
                        Develop algorithms for real-time adaptation to changing user preferences, ensuring dynamic and personalized movie recommendations.

                        **Interpretability:**\
                        Enhance model interpretability to provide users with clear insights into the rationale behind movie recommendations.

                        **User Feedback Integration:**\
                        Implement mechanisms for collecting and integrating user feedback to continually refine and improve recommendation algorithms based on user preferences.
                        """)   



if __name__ == "__main__":
    main()
