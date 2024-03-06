from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.mixture import GaussianMixture
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
import numpy as np
import faiss

def best_tree(df, test_size=0.35, diff=0.02):
    """ Function to run decisiontreeclassifier for varying max_depth
        Plots the results
        Prints the maximum accuracy and corresponding max_depth
        Prints the accuracy within {diff} of the maximum accuracy and the corresponding max_depth
        Inputs:
            - df: dataframe from songs csv file
            - test_size: desired test set size (<1)
            - diff: the difference between the maximum accuracy and "acceptable" accuracy
        Returns:
            - max_index_: max_depth correponding to the accuracy within diff of the maximum accuracy
            - max_index:  max_depth correponding to the maximum accuracy 
    """
    X, y = df.drop(['name', 'artist', 'id','genre','labels','description', 'lyrics', 'new description', 'combined description', 'duration_ms'], axis=1), df[['labels']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    accuracy = []
    depths = range(1,21)
    for depth in depths:
        clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
        clf = clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        a = accuracy_score(y_test, y_pred)
        accuracy.append(a)

    max_val, max_index = max(accuracy), accuracy.index(max(accuracy))+1
    max_index_ = max_index
    for idx, a in enumerate(accuracy[:max_index]):
        if max_val-a <= diff:
            max_index_ = idx+1
            break
    
    print(f"The maximum accuracy is {round(100*max_val,2)}% at max_depth={max_index}")
    print(f"The accuracy within {100*diff}% of the maximum accuracy is {round(100*accuracy[max_index_-1],2)}% at max_depth={max_index_}")
    data = {'X': depths, 'Y': accuracy}
    df = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='X', y='Y')
    plt.axvline(x=max_index_, linestyle='--', color='green', linewidth=1.5, label=f"Accuracy within {100*diff}% of maximum accuracy")
    plt.axvline(x=max_index, linestyle='--', color='red', linewidth=1.5, label="Maximum accuracy")
    plt.xticks(range(1,21))
    plt.title('Accuracy vs max_depth')
    plt.xlabel('max_depth')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    return max_index_, max_index

def describe_tree(clf, df, feature_names):
    """ Helper function to calculate the true value (not scaled values so ~N(0,1)) for each node's threshold value """
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
    while len(stack) > 0:
        node_id, depth = stack.pop()
        node_depth[node_id] = depth
        is_split_node = children_left[node_id] != children_right[node_id]
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True
    res = {}
    for i in range(n_nodes):
        if is_leaves[i]:
            pass
        else:
            var = feature_names[feature[i]]
            thres = threshold[i]
            if var not in res:
                res[var] = [thres]
            else:
                res[var].append(thres)
    for key in res.keys():
        l = df[key]
        u, std = np.mean(l), np.std(l)
        print(key)
        for val in res[key]: # z = (x - u) / s
            val_new = val*std + u
            print(f"     {round(val,2)}   {round(val_new,2)}")
    return res   
     
def run_tree(df, df_orig, n_cluster, max_depth=4, test_size=0.35):
    """ Function to run decisiontreeclassifier and plot the resulting tree 
        Inputs:
            - df: dataframe from GMM (standard scaler is already applied)
            - df_orig: dataframe from songs csv file
            - max_depth: max depth of decision tree
            - test_size: desired test set size (<1)
    """
    X, y = df.drop(['name', 'artist', 'id','genre','labels','description', 'lyrics', 'new description', 'combined description', 'duration_ms'], axis=1), df[['labels']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    train_accuracy = clf.score(X_train, y_train)
    print("Train accuracy: %.2f%%" % (train_accuracy * 100.0))
    accuracy = accuracy_score(y_test, y_pred)
    print("Test accuracy: %.2f%%" % (accuracy * 100.0))

    cm = confusion_matrix(y_test, y_pred)
    classes = list(range(0,len(cm)))
    songs_per_class = [sum(cm[i]) for i in range(len(cm))]
    class_acc = {"Class":classes, "Accuracy":[], "Accuracy [%]":[]}
    for i in range(len(cm)):
        num = cm[i][i]
        den = songs_per_class[i]
        class_acc["Accuracy"].append(f"{str(num)}/{str(den)}")
        class_acc["Accuracy [%]"].append(f"{round(100*num/den)}%")
    df = pd.DataFrame(class_acc)
    print("Cluster Accuracy:"); print(df.to_string(index=False))
    print("Confusion Matrix:"); print(cm)

    class_names = [str(i) for i in range(n_cluster)]
    feature_names = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature']
    plt.figure(figsize=(20, 10))
    tree.plot_tree(clf, feature_names=feature_names, class_names=class_names, precision=2, rounded=True, fontsize=9)
    plt.show()

def best_GMM(songs_file):
    """ Function to find the best model (according to lowest BIC score) for varying covariance types and number of components.
        Prints the best hyperparameters and returns the best model.
        Inputs:
            - songs_file: csv file with song descriptions
    """
    def gmm_bic_score(estimator, X):
        return -estimator.bic(X)
    df = pd.read_csv(songs_file)
    param_grid = {"n_components": range(1, 50),
                  "covariance_type": ["spherical", "tied", "diag", "full"]}
    grid_search = GridSearchCV(GaussianMixture(random_state=42), param_grid=param_grid, scoring=gmm_bic_score)
    numerical_columns = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature']
    scaler = StandardScaler()
    X = df[numerical_columns]
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    grid_search.fit(X)
    best_gmm_model = grid_search.best_estimator_
    print("Best Hyperparameters:", grid_search.best_params_)
    best_gmm_model.fit(X)
    return best_gmm_model

def GMM(songs_file, n_clusters):
    """ Function to run Gaussian Mixture Models (GMM) using strictly tabular data (i.e. Spotify audio features)
        Returns the updated dataframe with the "labels" column
        Inputs:
            - songs_file: csv file with song descriptions
            - n_clusters: number of parameters
    """
    df = pd.read_csv(songs_file)
    numerical_columns = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature']
    scaler = StandardScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    gmm = GaussianMixture(n_components=n_clusters, covariance_type="diag", random_state=42)
    model = gmm.fit(df[numerical_columns])
    labels = gmm.predict(df[numerical_columns])
    df['labels'] = labels
    bic_score = model.bic(df[numerical_columns])
    print(f"\n\nAUDIO FEATURES:\nThe GMM model with {n_clusters} clusters has a BIC score of {round(bic_score,0)}.")
    return df

def best_GMM_string(description):
    """ Function to find the best model (according to lowest BIC score) for varying covariance types and number of components.
        Assumes associated .npy file is in the same directory. If user does not have .npy file, then call create_embeddings to create .npy files
        Prints the best hyperparameters and returns the best model.
        Inputs:
            - description: the column header of one of the three descriptions: "description", "new description", "combined description"
    """
    def gmm_bic_score(estimator, X):
        return -estimator.bic(X)    
    file = description + ' encoding.npy'
    try:
        data = np.load(file)
        df = pd.DataFrame(data)
    except:
        print(f"Error: {file} not found")
        return None
    param_grid = {"n_components": range(1, 30), "covariance_type": ["spherical", "tied", "diag", "full"]}
    grid_search = GridSearchCV(GaussianMixture(random_state=42), param_grid=param_grid, scoring=gmm_bic_score)
    grid_search.fit(data)
    best_gmm_model = grid_search.best_estimator_
    print("Best Hyperparameters:", grid_search.best_params_)
    best_gmm_model.fit(data)
    return best_gmm_model

def GMM_string(songs_file, description, n_clusters):
    """ Function to run Gaussian Mixture Models (GMM) using the text embeddings of the songs
        Returns the updated dataframe with the "labels" column
        Inputs:
            - songs_file: csv file with song descriptions
            - description: the column header of one of the three descriptions: "description", "new description", "combined description"
            - n_clusters: number of parameters
    """
    assert description in ["description", "new description", "combined description"]
    df =  pd.read_csv(songs_file)
    file = description + ' encoding.npy'
    try:
        data = np.load(file)
    except:
        print(f"Error: {file} not found")
        return None
    
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    model = gmm.fit(data)
    labels = gmm.predict(data)
    
    df['labels'] = labels
    bic_score = model.bic(data)
    print(f"\n\nSONG EMBEDDINGS:\nThe GMM model with {n_clusters} clusters has a BIC score of {round(bic_score,0)}.")
    df.to_csv("clusters.csv", index=False)  
    return df 

def create_embeddings(songs_file):
    """ Function to create embeddings for all songs based on the three 'description' inputs 
        Saves numpy array to .npy file for quick access
        Inputs:
            - songs_file: name of csv file for song descriptions
    """
    from sentence_transformers import SentenceTransformer
    df_songs = pd.read_csv(songs_file)
    for description in ["description", "new description", "combined description"]:
        model = SentenceTransformer('all-mpnet-base-v2')
        song_descs = df_songs[description]
        vectors = model.encode(song_descs)
        np.save(description + ' encoding.npy', vectors)
        print("Done")

def podcast_song_matching(podcasts_file, songs_file):
    """ Function to recommend the top three songs with highest cossine similarity for each podcast episode 
        Inputs:
            - podcasts_file: name of csv file for posdcast episodes
            - songs_file: name of csv file for song descriptions
    """
    from sentence_transformers import SentenceTransformer
    pods = {"description":[], "new description":[], "combined description":[]}
    df_pod, df_songs = pd.read_csv(podcasts_file), pd.read_csv(songs_file)
    podcasts = df_pod["Description"]
    songs, artists = df_songs["name"], df_songs["artist"]
    
    for description in pods.keys():
        model = SentenceTransformer('all-mpnet-base-v2')
        song_descs = df_songs[description]
        vectors = model.encode(song_descs)
        vector_dimension = vectors.shape[1]
        index = faiss.IndexFlatL2(vector_dimension)
        index.add(vectors)

        for idx in range(len(podcasts)):
            if idx % 50 == 0: print(idx)
            search_text = podcasts[idx]
            search_vector = model.encode(search_text)
            _vector = np.array([search_vector])
            dist, ann = index.search(_vector, k=3)
            dist, ann = dist[0].tolist(), ann[0].tolist()
            s = ""
            for idx, i in enumerate(ann): s += f"{songs[i]} by {artists[i]}: {round(dist[idx],2)}\n"
            pods[description].append(s)
        
    df2 = pd.DataFrame.from_dict(pods)
    df = pd.concat([df_pod, df2], axis=1)
    df.to_csv("podcasts_songs.csv", index=False)  

def plot_genres_clusters(df, n_cluster):
    """ Function to create bar plots for genres in each GMM group
        Inputs:
            - df: dataframe with "labels" column (from GMM)
            - n_cluster: number of groups
    """
    songs  = {key: [] for key in list(range(n_cluster))}
    genres = {key: {} for key in list(range(n_cluster))}
    for index, row in df.iterrows():
        genre, label, song = row["genre"], row["labels"], row["name"]
        songs[label].append(song) 
        if genre not in genres[label]: 
            genres[label][genre] = 1
        else:
            genres[label][genre] += 1

    reshaped_data = []
    for key in genres:
        values = list(genres[key].values())
        labels = list(genres[key].keys())
        df = pd.DataFrame({'Label': labels, 'Value': values, 'Key': key})
        reshaped_data.append(df)
    df_concat = pd.concat(reshaped_data)
    plt.figure(figsize=(15, 6))
    ax = sns.barplot(data=df_concat, x='Key', y='Value', hue="Label")
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))    
    plt.title('Genres per Group'); plt.xlabel('Groups'); plt.ylabel('Count')
    plt.show()

def song_counts(podcast_song_file, songs_file, description, song_name, cutoff = 0):
    """ Function to return the number of times a song is suggested for a podcast. 
        Returns the description of the song and a dataframe of the podcasts and associated counts
        Inputs:
            - podcast_song_file: name of csv file containing song recommendations for podcasts (podcasts_songs.csv)
            - songs_file: name of csv file for song descriptions
            - description: the column header of one of the three descriptions: "description", "new description", "combined description"
            - song_name: name of the song of interest
            - cutoff: cutoff of number of times song appears (default is set to 0). Count must be > cutoff to be passed into the final dataframe
    """
    assert description in ["description", "new description", "combined description"]

    song_name = song_name.lower()
    df_song = pd.read_csv(songs_file)
    result = df_song[df_song['name'].str.lower() == song_name.lower()]
    if not result.empty: d = result[description].iloc[0]
    else: d = None

    df = pd.read_csv(podcast_song_file)
    pod = {}
    
    podcasts, description = df["Podcast Name"], df[description]
    for idx, desc in enumerate(description):
        if song_name in desc.lower():
            podcast = podcasts[idx]
            if podcast not in pod:
                pod[podcast] = [1]
            else:
                pod[podcast] = [pod[podcast][0] + 1]
    if cutoff > 0:
        pod_new = {}
        for key in pod.keys():
            val = pod[key][0]
            if val > cutoff:
                pod_new[key] = val
        pod = pod_new
    pod = dict(sorted(pod.items(), key=lambda item: item[1]))
    df = pd.DataFrame.from_dict(pod, orient="index")
    return d, df

def visualize_genres(songs_file):
    """ Function to create bar plots for genres across all songs (sorted in ascending order)
        Inputs:
            - songs_file: name of csv file for song descriptions
    """
    df = pd.read_csv(songs_file)
    genre_counts = df['genre'].value_counts().reset_index()
    genre_counts.columns = ['genre', 'count']
    genre_counts_sorted = genre_counts.sort_values(by='count')
    plt.figure(figsize=(12, 6))
    sns.barplot(data=genre_counts_sorted, x='genre', y='count')
    plt.title('Song Genres')
    plt.xlabel('Genre')
    plt.ylabel('Count')
    plt.show()


n_cluster = 5
description = "new description"
songs_file, podcasts_file, podcast_song_file = "songs.csv", "podcasts.csv", "podcasts_songs.csv"
df_orig = pd.read_csv(songs_file)

# visualize_genres(songs_file) 
###################################################  BELOW: GAUSSIAN MIXTURE MODELS (GMM) ################################################### 
# BELOW: GMM using Spotify's audio features 
df = GMM(songs_file, n_cluster)                                      # runs GMM  
max_depth, max_depth_best = best_tree(df, test_size=0.2, diff=0.03)  # find max_depth with accuracy within {diff} of the maximum accuracy
run_tree(df, df_orig, n_cluster, max_depth=max_depth, test_size=0.35)           # creates decision tree using labels from GMM & prints confusion matrix
plot_genres_clusters(df, n_cluster)                                  # creates bar plots for genres in each GMM group

###################################################  BELOW: PODCAST RECOMMENDATIONS ################################################### 
# podcast_song_matching(podcasts_file, songs_file) # generates song recommendations based on podcast episode descriptions

# BELOW: call to count the number of times a specific song is suggested for each podcast
# song_name = "long live" # change song of interest
# desc, df = song_counts(podcast_song_file, songs_file, description, song_name, cutoff = 3)
# print(desc); print(df)

###################################################  BELOW: SONG EMBEDDINGS ################################################### 
# create_embeddings(songs_file) # creates embeddings for all songs based on the three 'description' inputs: "description", "new description", "combined description"
