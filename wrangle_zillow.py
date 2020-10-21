
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.preprocessing
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

from env import host, password, user 
    

##### ACQUIRE #####

def get_connection(db, user=user, host=host, password=password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

def zillow_datac():
    
    sql_query = '''
    SELECT *
FROM properties_2017 as prop 
INNER JOIN (
    SELECT id, p.parcelid, logerror, transactiondate
    FROM predictions_2017 AS p
    INNER JOIN (
    SELECT parcelid,  MAX(transactiondate) AS max_date
    FROM predictions_2017 
    GROUP BY (parcelid)) AS sub
        ON p.parcelid = sub.parcelid
    WHERE p.transactiondate = sub.max_date) AS subq
    ON prop.id = subq.id
    where latitude and longitude is not null
        '''
    df = pd.read_sql(sql_query, get_connection('zillow'))
    df.to_csv('zillow')
    return df

##### PREPARE #####

def get_counties(df):
    county_df = pd.get_dummies(df.fips)
    # rename columns by actual county name
    county_df.columns = ['LA', 'Orange', 'Ventura']
    # concatenate the dataframe with the 3 county columns to the original dataframe
    df_dummies = pd.concat([df, county_df], axis = 1)
    # drop regionidcounty and fips columns
    df = df_dummies.drop(columns = ['fips'])

    df = df.drop(columns=['parcelid','typeconstructiontypeid','storytypeid','propertylandusetypeid',
                      'heatingorsystemtypeid','buildingclasstypeid','architecturalstyletypeid',
                      'airconditioningtypeid','id','basementsqft','buildingqualitytypeid','decktypeid',
                      'finishedfloor1squarefeet','finishedsquarefeet12','finishedsquarefeet13',
                      'finishedsquarefeet50','finishedsquarefeet15','finishedsquarefeet6','fireplacecnt',
                      'garagetotalsqft','poolsizesum','pooltypeid10','pooltypeid2','pooltypeid7',
                      'propertycountylandusecode','propertyzoningdesc','rawcensustractandblock',
                      'regionidcounty','regionidneighborhood','threequarterbathnbr','unitcnt',
                      'yardbuildingsqft17','yardbuildingsqft26','fireplaceflag',
                      'assessmentyear','taxdelinquencyflag','taxdelinquencyyear','id',])
    return df

def create_features(df):
    df['age'] = 2017 - df.yearbuilt

    # create taxrate variable
    df['taxrate'] = df.taxamount/df.taxvaluedollarcnt

    #changing the values to something a bit more graphable
    df.latitude = df.latitude / 10000000
    df.longitude = df.longitude / 10000000
    
    # create acres variable
    df['acres'] = df.lotsizesquarefeet/43560
    
    # dollar per square foot-structure
    df['structure_dollar_per_sqft'] = df.structuretaxvaluedollarcnt/df.calculatedfinishedsquarefeet

    # dollar per square foot-land
    df['land_dollar_per_sqft'] = df.landtaxvaluedollarcnt/df.lotsizesquarefeet
    
    # ratio of beds to baths
    df['bed_bath_ratio'] = df.bedroomcnt/df.bathroomcnt
    
    # 12447 is the ID for city of LA. 
    # I confirmed through sampling and plotting, as well as looking up a few addresses.
    df['cola'] = df['regionidcity'].apply(lambda x: 1 if x == 12447.0 else 0)

    df = df.drop(columns=['calculatedbathnbr','fullbathcnt','garagecarcnt','hashottuborspa','regionidcity',
                 'regionidzip','numberofstories','censustractandblock','cola','transactiondate','poolcnt'])
    
    return df

def remove_outliers(df):
    '''
    remove outliers in bed, bath, zip, square feet, acres & tax rate
    '''

    return df[((df.bathroomcnt <= 7) & (df.bedroomcnt <= 7) &  
               (df.bathroomcnt > 0) & 
               (df.bedroomcnt > 0) & 
               (df.acres < 10) &
               (df.calculatedfinishedsquarefeet < 7000) & 
               (df.taxrate < .05)
              )]    

def split_scale(df):
    train_validate, test = train_test_split(df, test_size = .2, random_state = 123)
    train, validate = train_test_split(train_validate, test_size = .3, random_state = 123)
    
    X_train = train.drop(columns=['logerror'])
    X_validate = validate.drop(columns=['logerror'])
    X_test = test.drop(columns=['logerror'])
    train = train

    y_train = train[['logerror']]
    y_validate = validate[['logerror']]
    y_test = test[['logerror']]
    
#def minmax_scaler(df)
    # create the scaler object and fit to X_train (get the min and max from X_train for each column)
    scaler = MinMaxScaler(copy=True, feature_range=(0,1)).fit(X_train)

    # transform X_train values to their scaled equivalent and create df of the scaled features
    X_train_scaled = pd.DataFrame(scaler.transform(X_train), 
                                  columns=X_train.columns.values).set_index([X_train.index.values])
    
    # transform X_validate values to their scaled equivalent and create df of the scaled features
    X_validate_scaled = pd.DataFrame(scaler.transform(X_validate),
                                    columns=X_validate.columns.values).set_index([X_validate.index.values])

    # transform X_test values to their scaled equivalent and create df of the scaled features   
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), 
                                 columns=X_test.columns.values).set_index([X_test.index.values])

    scaler_train = MinMaxScaler(copy=True, feature_range=(0,1)).fit(train)
    # transform X_train values to their scaled equivalent and create df of the scaled features
    train_scaled = pd.DataFrame(scaler_train.transform(train), 
                                    columns=train.columns.values).set_index([train.index.values])

    
    return X_train, X_validate, X_test, train, train_scaled, y_train, y_validate, y_test, X_train_scaled, X_validate_scaled, X_test_scaled
    

##### EXPLORE #####

def elbow_plot(X_train_scaled, cluster_vars):
    # elbow method to identify good k for us
    ks = range(2,20)
    
    # empty list to hold inertia (sum of squares)
    sse = []

    # loop through each k, fit kmeans, get inertia
    for k in ks:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X_train_scaled[cluster_vars])
        # inertia
        sse.append(kmeans.inertia_)

    print(pd.DataFrame(dict(k=ks, sse=sse)))

    # plot k with inertia
    plt.plot(ks, sse, 'bx-')
    plt.xlabel('k')
    plt.ylabel('SSE')
    plt.title('Elbow method to find optimal k')
    plt.show()


def run_kmeans(X_train, X_train_scaled, k, cluster_vars, cluster_col_name):
    
    # create kmeans object
    kmeans = KMeans(n_clusters = k, random_state = 13)
    kmeans.fit(X_train_scaled[cluster_vars])
    # predict and create a dataframe with cluster per observation
    train_clusters = \
        pd.DataFrame(kmeans.predict(X_train_scaled[cluster_vars]),
                              columns=[cluster_col_name],
                              index=X_train.index)
    
    return train_clusters, kmeans

def add_to_train(train_clusters, centroids, X_train, X_train_scaled, cluster_col_name):
    # concatenate cluster id
    X_train2 = pd.concat([X_train, train_clusters], axis=1)

    # join on clusterid to get centroids
    X_train2 = X_train2.merge(centroids, how='left', 
                            on=cluster_col_name).\
                        set_index(X_train.index)
    
    # concatenate cluster id
    X_train_scaled2 = pd.concat([X_train_scaled, train_clusters], 
                               axis=1)

    # join on clusterid to get centroids
    X_train_scaled2 = X_train_scaled2.merge(centroids, how='left', 
                                          on=cluster_col_name).\
                            set_index(X_train.index)
    
    return X_train2, X_train_scaled2

def kmeans_transform(X_scaled, kmeans, cluster_vars, cluster_col_name):
    kmeans.transform(X_scaled[cluster_vars])
    trans_clusters = \
        pd.DataFrame(kmeans.predict(X_scaled[cluster_vars]),
                              columns=[cluster_col_name],
                              index=X_scaled.index)
    
    return trans_clusters



def get_centroids(cluster_vars, cluster_col_name, kmeans):
    centroid_col_names = ['centroid_' + i for i in cluster_vars]

    centroids = pd.DataFrame(kmeans.cluster_centers_, 
             columns=centroid_col_names).reset_index().rename(columns={'index': cluster_col_name})
    
    return centroids
