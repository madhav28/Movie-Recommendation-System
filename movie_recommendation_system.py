import streamlit as st
import pandas as pd

if 'user_id' not in st.session_state:
    st.session_state.user_id = None


def login(username, password):
    return username >= 1 and username <= 162541


def main():

    is_logged_in = st.session_state.get("is_logged_in", False)

    if not is_logged_in:
        st.title("Movie Recommendation System")

        username = st.text_input(
            "Enter a User ID from 1 to 162541:", value="9")
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
                st.error("Please enter a User ID from 1 to 162541!")
    else:
        col1, col2 = st.columns([4, 1])

        if col2.button("Logout"):
            st.session_state.is_logged_in = False
            st.experimental_rerun()
        app_content()


def app_content():
    user_id = st.session_state.user_id

    st.title("Movie Recommendation System")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(['Introduction', 'Dataset', 'Data Visualisation',
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
        st.markdown("""Three datasets are used in this project: ratings.csv, movies.csv and tags.csv. 
                    The features of each dataset are described below:""")
        st.image("data_and_features.png")
        st.markdown("""The tags.csv file was used only to develop a content-based NLP recommendation system. 
                    The ratings.csv and movies.csv files were used to develop all other recommendation systems.""")

    with tab3:
        st.markdown("#### Distributions for Movies:")
        st.image("movie_release.png")
        st.image("movie_genres.png")
        st.markdown("#### Distributions for Movie Ratings:")
        st.image("average_rating.png")
        st.image("movie_rating_rel.png")
        st.image("movie_rating_genre.png")
        st.image("most_rated_movies.png")

    with tab4:
        tab_a1, tab_a2, tab_a3 = st.tabs(["Collaborative Filtering",
                                          "Content-Based Filtering", "Hybrid"])
        with tab_a1:
            st.markdown("""
                        Collaborative filtering is a technique used in recommendation systems to predict a user's preferences or interests by leveraging the preferences and behaviors of a group of users. The underlying idea is that users who have agreed in the past on certain issues tend to agree again in the future. This method assumes that if a user A has similar preferences to a user B on a certain issue, A is more likely to share B's preferences on a different issue as well.

            **User-Based Collaborative Filtering:**

            This approach recommends items to a target user based on the preferences and behavior of users who are similar to that target user.
            The system identifies users with similar preferences to the target user and recommends items that those similar users have liked or interacted with.
            The similarity between users is often calculated using different metrics such as cosine similarity or Pearson correlation.
            Item-Based Collaborative Filtering:

            In this approach, the system recommends items similar to those that a user has liked or interacted with in the past.
            It identifies items that are similar to the ones the target user has shown interest in and recommends them.
            Similarity between items is also calculated using metrics like cosine similarity or Pearson correlation.
                        """)

            tab_a1_mat, tab_a1_dl = st.tabs(
                ["Matrix Factorization", "Deep Learning Based"])
            with tab_a1_mat:
                st.markdown("#### Matrix Factorization:")
                st.markdown("""Matrix factorization stands out as a widely used technique for collaborative filtering. 
                            The fundamental concept involves breaking down the user-item matrix into two matrices with lower ranks: one depicting user preferences and the other reflecting movie characteristics. 
                            The reconstruction of the original user-item matrix is accomplished by computing the dot product of these two matrices.""")
                st.markdown("""The user-item matrix features rows corresponding to users and columns corresponding to movies. 
                            The matrix entries denote the ratings provided by users for the respective movies. 
                            Typically, this matrix is sparse, given that each user tends to rate only a small portion of the available movies.""")
                st.image("matrix_fact_img.png")
                st.markdown("""The goal of matrix factorization is to decompose the user-item matrix R into two lower-rank matrices, P and Q, with their product closely approximating R. 
                            Matrix P signifies user’s preferences, while matrix Q signifies items' characteristics. 
                            Each row in matrix P is a vector representing a user's preferences, and each row in matrix Q is a vector representing an item's characteristics. 
                            This approach allows for a meaningful representation of user-item interactions in terms of preferences and characteristics.""")
                st.markdown("""This method aims to discover matrices P and Q that minimize the discrepancy between the approximated matrix and the real matrix. 
                            Typically, this disparity is gauged through the root mean squared error (RMSE) between the predicted ratings and the actual ratings.""")
                st.markdown("""In this project, the SVD (Singular Value Decomposition) technique is incorporated for matrix factorization using the Surprise library in Python. 
                            SVD is a linear algebra technique used for matrix factorization. 
                            In the context of collaborative filtering, it helps decompose the user-item matrix into three matrices (P, Σ, Q^T), where P represents user preferences, Σ is a diagonal matrix of singular values, and Q^T represents movie characteristics.""")
                st.markdown("---")
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
                st.markdown("#### Deep Learning Based:")
                st.markdown("""Deep Neural Network (DNN) models provide a promising solution for addressing challenges like the cold start problem and improving the relevance of recommendations. 
                            The flexibility of the input layer in DNNs allows for the incorporation of user and movie features, enabling a more nuanced understanding of user preferences and enhancing the relevance of recommendations.""")
                st.markdown("""In this project, Deep Neural Networks are employed for movie recommendations. 
                            Users and movies undergo one-hot encoding and are then input into the Deep Neural Network as distinct inputs, with the ratings being generated as the output.""")
                st.markdown("""The construction of the Deep Neural Network model involved extracting the latent features of users and movies using Embedding layers. 
                            Subsequently, Dense layers with dropout mechanisms were stacked, followed by the addition of a final Dense layer comprising 9 neurons (representing each rating from 1 to 5) and incorporating a Softmax activation function. 
                            For the optimization algorithm, it was decided to use SGD and Sparse Categorical Cross Entropy for the loss function.""")
                st.image("dl_img.png")
                st.markdown("""The workflow involves users inputting their ID, extracting unseen movies, and utilizing a DNN model that takes user and movie IDs to predict ratings. 
                         The model's predictions are normalized and used to identify movies likely to interest the user, without converting ratings back to the original scale. 
                         The DNN model is thus a valuable tool for predicting user preferences for unseen movies.""")
                st.markdown("---")
                if user_id >= 1 and user_id <= 10:
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
                else:
                    st.error(
                        """Due to limitations on computational resources, movie recommendations were generated for only the first 10 users using this model. 
                        Please login using a User ID from 1 to 10 to view the recommendations provided by this model.""")

        with tab_a2:
            st.markdown("""
                        Content-based filtering is another approach used in recommendation systems, and it relies on the characteristics or features of items and users to make recommendations. Unlike collaborative filtering, content-based filtering doesn't require information about the preferences or behaviors of other users. Instead, it focuses on the properties of items and the explicit profile of the user.
                        """)

            st.markdown("""

                        ### Content-Based Prediction Using NLP:

                        **Word2Vec Training:**\
                        Tokenize movie tags and train a Word2Vec model with specified parameters, such as vector size and window size. This process generates embeddings that capture semantic relationships within the textual content.

                        **Movie Embeddings:**\
                        Compute unique embeddings for each movie using the trained Word2Vec model. These embeddings represent the semantic content of the movies.

                        **User Embeddings:**\
                        Determine user embeddings by aggregating the embeddings of rated movies, reflecting individual preferences in the Word2Vec space.

                        **Cosine Similarity for Recommendations:**\
                        Utilize cosine similarity to measure the similarity between user and movie embeddings.
                        Recommend movies with the highest cosine similarity scores, aligning with user preferences.
                        
                        """)
            st.image("NLP_based_method.png")
            st.markdown("---")
            if user_id >= 1 and user_id <= 100:
                st.markdown("#### Top 10 Recommendations (for User ID = " +
                            str(user_id)+") are:  ")
                content_nlp = pd.read_csv("content_nlp.csv")
                content_nlp = content_nlp[content_nlp['user_id'] == user_id]
                recommendations = {}
                top_10 = []
                for column in content_nlp.columns:
                    if column == 'user_id':
                        continue
                    top_10.append(content_nlp[column].ravel()[0])

                recommendations = {"Movie Recommendation": top_10}
                recommendations = pd.DataFrame(recommendations)
                recommendations.set_index(pd.RangeIndex(
                    start=1, stop=11, step=1), inplace=True)
                recommendations = recommendations.rename_axis('#')

                st.dataframe(recommendations)
            else:
                st.error(
                    """Due to limitations on computational resources, movie recommendations were generated for only the first 100 users using this model. 
                        Please login using a User ID from 1 to 100 to view the recommendations provided by this model.""")

        with tab_a3:
            st.markdown("#### Hybrid Recommendation System:")
            st.markdown("""
                        A hybrid recommendation system is an approach that combines multiple recommendation techniques to overcome the limitations of individual methods and provide more accurate and diverse recommendations. By leveraging the strengths of different recommendation algorithms, hybrid models aim to enhance overall performance and address challenges such as the cold start problem, sparsity of data, and the diversity of recommendations. 
                        """)
            st.markdown("#### Methodology:")
            st.markdown("""Hybrid recommendation is developed as an ensemble model of item-based and user-based recommendation systems. 
                        To develop these recommendation systems, a data matrix with rows as user, columns as movies and values as ratings is constructed. 
                        In this model, both item-based and user-based similarities are used for movie recommendations. 
                        The approaches used for identifying item-based similarities and user-based similarities are described below.""")
            st.markdown("""For **item-based** similarity, initially, the user's most recent favorite movie is identified. 
                        Later, using the data matrix, movies which are strongly correlated with this favorite movie are identified as potential recommendations to the user. """)
            st.markdown("""For user-based similarity, initially, a list of movies rated by the user is found. 
                        Next, all the users who rated at least 60% (hyperparameter) of the previous list of movies and who have a correlation of at least 0.75 are identified. 
                        After that, this list of similar users is used to estimate the weighted average rating of the movies. 
                        Finally, top movies with the highest weighted average rating are identified as the potential recommendations to the user.""")
            st.markdown("---")
            if user_id >= 1 and user_id <= 10:
                st.markdown("#### Top 10 Recommendations (for User ID = " +
                            str(user_id)+") are:  ")
                hybrid = pd.read_csv("hybrid.csv")
                hybrid = hybrid[hybrid['user_id'] == user_id]
                recommendations = {}
                top_10 = []
                for column in hybrid.columns:
                    if column == 'user_id':
                        continue
                    top_10.append(hybrid[column].ravel()[0])

                recommendations = {"Movie Recommendation": top_10}
                recommendations = pd.DataFrame(recommendations)
                recommendations.set_index(pd.RangeIndex(
                    start=1, stop=11, step=1), inplace=True)
                recommendations = recommendations.rename_axis('#')

                st.dataframe(recommendations)
            else:
                st.error(
                    """Due to limitations on computational resources, movie recommendations were generated for only the first 10 users using this model. 
                        Please login using a User ID from 1 to 10 to view the recommendations provided by this model.""")

        with tab5:
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

        with tab6:
            st.markdown("""
                        ## Conclusion:
                        In conclusion, the project has successfully developed and implemented a robust movie prediction system, leveraging collaborative filtering and Content-Based filtering techniques. The evaluation results showcase the effectiveness of deep learning models, closely followed by matrix factorization, in accurately predicting user ratings for movies. The algorithm has demonstrated the ability to provide recommendations that align well with human suggestions.

                        The incorporation of features such as similar movies, genres, and popularity has enhanced the algorithm's recommendation capabilities. The decision to train the models on the entire dataset has proven fruitful, contributing to improved accuracy on the test set.

                        The project marks a significant step forward in personalized movie recommendations, creating a user-friendly application that harnesses the power of collaborative filtering to enhance the movie-watching experience.

                        ## Future Works:
                        While the current models have shown promising results, there are several avenues for future improvements and enhancements:

                        **Incorporating temporal factors:**\
                        Considering how movie trends and audience preferences evolve over time. By accounting for temporal aspects such as changing graphics quality or narrative styles across decades, the system could further refine its recommendations, ensuring they remain relevant and appealing to users amidst evolving cinematic landscapes.
                        
                        **Interpretability:**\
                        Enhance model interpretability to provide users with clear insights into the rationale behind movie recommendations.

                        **User Feedback Integration:**\
                        Implement mechanisms for collecting and integrating user feedback to continually refine and improve recommendation algorithms based on user preferences.
                        """)


if __name__ == "__main__":
    main()
