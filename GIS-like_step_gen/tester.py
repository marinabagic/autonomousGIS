# import geopandas as gpd
# import pandas as pd
# from shapely.geometry import Point
# from shapely.ops import nearest_points
# import matplotlib.pyplot as plt

# import geopandas as gpd

# def load_sc_zip_boundary(sc_zip_boundary_url):
#     sc_zip_boundary_gdf = gpd.read_file(sc_zip_boundary_url)
#     return sc_zip_boundary_gdf


# import geopandas as gpd
# import pandas as pd
# from shapely.geometry import Point
# from shapely.ops import nearest_points

# def load_sc_hospitals(sc_hospitals_url):
#     sc_hospitals_df = pd.read_csv(sc_hospitals_url)
#     sc_hospitals_df = sc_hospitals_df.dropna(subset=['POINT_X', 'POINT_Y'])
#     sc_hospitals_df['POINT_X'] = sc_hospitals_df['POINT_X'].astype(float)
#     sc_hospitals_df['POINT_Y'] = sc_hospitals_df['POINT_Y'].astype(float)
#     sc_hospitals_points = [Point(x, y) for x, y in zip(sc_hospitals_df['POINT_X'], sc_hospitals_df['POINT_Y'])]
#     sc_hospitals_gdf = gpd.GeoDataFrame(sc_hospitals_df, geometry=sc_hospitals_points)
#     sc_hospitals_gdf.crs = "EPSG:4326"
#     sc_hospitals_gdf = sc_hospitals_gdf.to_crs("EPSG:3857")

#     return sc_hospitals_gdf


# def calculate_centroid(sc_zip_boundary_gdf):
#     sc_zip_boundary_gdf_with_centroid = sc_zip_boundary_gdf.copy()
#     sc_zip_boundary_gdf_with_centroid['centroid'] = sc_zip_boundary_gdf_with_centroid.geometry.centroid
#     return sc_zip_boundary_gdf_with_centroid


# import geopandas as gpd
# from shapely.geometry import Point

# def create_points_from_hospitals(sc_hospitals_df):
#     sc_hospitals_points = [Point(x, y) for x, y in zip(sc_hospitals_df['POINT_X'], sc_hospitals_df['POINT_Y'])]
#     return sc_hospitals_points


# def calculate_nearest_hospital(sc_zip_boundary_gdf_with_centroid, sc_hospitals_points):
#     sc_zip_boundary_gdf_with_centroid['nearest_hospital'] = sc_zip_boundary_gdf_with_centroid.geometry.apply(lambda x: nearest_points(x, sc_hospitals_points)[1])
#     return sc_zip_boundary_gdf_with_centroid


# import geopandas as gpd
# import pandas as pd
# from shapely.geometry import Point
# from shapely.ops import nearest_points

# def calculate_distance(sc_zip_boundary_gdf_with_nearest_hospital):
#     sc_zip_boundary_gdf_with_nearest_hospital['distance'] = sc_zip_boundary_gdf_with_nearest_hospital.apply(lambda row: row['geometry'].distance(row['nearest_hospital']), axis=1)
#     return sc_zip_boundary_gdf_with_nearest_hospital


# import geopandas as gpd
# import pandas as pd
# from shapely.geometry import Point
# from shapely.ops import nearest_points
# import matplotlib.pyplot as plt

# def create_choropleth_map(sc_zip_boundary_gdf_with_distance):
#     sc_zip_boundary_gdf_with_distance.plot(column='distance', cmap='viridis', legend=True, figsize=(15, 10))
#     plt.title('Distance to Nearest Hospital')
#     plt.show()

# def assembely_solution():
#     sc_zip_boundary_url = "https://github.com/GIBDUSC/test/raw/master/sc_zip_boundary.zip"
#     sc_zip_boundary_gdf = gpd.read_file(sc_zip_boundary_url)

#     sc_hospitals_url = "https://github.com/gladcolor/spatial_data/raw/master/South_Carolina/SC_hospitals_with_emergency_room_cleaned.csv"
#     sc_hospitals_df = pd.read_csv(sc_hospitals_url)
#     sc_hospitals_df = sc_hospitals_df.dropna(subset=['POINT_X', 'POINT_Y'])
#     sc_hospitals_df['POINT_X'] = sc_hospitals_df['POINT_X'].astype(float)
#     sc_hospitals_df['POINT_Y'] = sc_hospitals_df['POINT_Y'].astype(float)
#     sc_hospitals_points = [Point(x, y) for x, y in zip(sc_hospitals_df['POINT_X'], sc_hospitals_df['POINT_Y'])]
#     sc_hospitals_gdf = gpd.GeoDataFrame(sc_hospitals_df, geometry=sc_hospitals_points)
#     sc_hospitals_gdf.crs = "EPSG:4326"
#     sc_hospitals_gdf = sc_hospitals_gdf.to_crs("EPSG:3857")

#     sc_zip_boundary_gdf_with_centroid = sc_zip_boundary_gdf.copy()
#     sc_zip_boundary_gdf_with_centroid['centroid'] = sc_zip_boundary_gdf_with_centroid.geometry.centroid

#     sc_zip_boundary_gdf_with_nearest_hospital = sc_zip_boundary_gdf_with_centroid.copy()
#     sc_zip_boundary_gdf_with_nearest_hospital['nearest_hospital'] = sc_zip_boundary_gdf_with_nearest_hospital.geometry.apply(lambda x: nearest_points(x, sc_hospitals_points)[1])

#     sc_zip_boundary_gdf_with_distance = sc_zip_boundary_gdf_with_nearest_hospital.copy()
#     sc_zip_boundary_gdf_with_distance['distance'] = sc_zip_boundary_gdf_with_distance.apply(lambda row: row['geometry'].distance(row['nearest_hospital']), axis=1)

#     sc_zip_boundary_gdf_with_distance.plot(column='distance', cmap='viridis', legend=True, figsize=(15, 10))
#     plt.title('Distance to Nearest Hospital')
#     plt.show()

#     return sc_zip_boundary_gdf_with_distance

# if __name__ == '__main__':
#     assembely_solution()

# import geopandas as gpd
# import pandas as pd
# from shapely.geometry import Point
# from shapely.ops import nearest_points
# import matplotlib.pyplot as plt

# def load_sc_zip_boundary(sc_zip_boundary_url):
#     sc_zip_boundary_gdf = gpd.read_file(sc_zip_boundary_url)
#     return sc_zip_boundary_gdf

# def load_sc_hospitals(sc_hospitals_url):
#     sc_hospitals_df = pd.read_csv(sc_hospitals_url)
#     sc_hospitals_df = sc_hospitals_df.dropna(subset=['POINT_X', 'POINT_Y'])
#     sc_hospitals_df['POINT_X'] = sc_hospitals_df['POINT_X'].astype(float)
#     sc_hospitals_df['POINT_Y'] = sc_hospitals_df['POINT_Y'].astype(float)
#     sc_hospitals_points = [Point(x, y) for x, y in zip(sc_hospitals_df['POINT_X'], sc_hospitals_df['POINT_Y'])]
#     sc_hospitals_gdf = gpd.GeoDataFrame(sc_hospitals_df, geometry=sc_hospitals_points)
#     sc_hospitals_gdf.crs = "EPSG:4326"
#     sc_hospitals_gdf = sc_hospitals_gdf.to_crs("EPSG:3857")
#     return sc_hospitals_gdf

# def calculate_centroid(sc_zip_boundary_gdf):
#     sc_zip_boundary_gdf = sc_zip_boundary_gdf.to_crs("EPSG:3857")
#     sc_zip_boundary_gdf_with_centroid = sc_zip_boundary_gdf.copy()
#     sc_zip_boundary_gdf_with_centroid['centroid'] = sc_zip_boundary_gdf_with_centroid.geometry.centroid
#     return sc_zip_boundary_gdf_with_centroid

# def create_points_from_hospitals(sc_hospitals_gdf):
#     sc_hospitals_points = list(sc_hospitals_gdf.geometry)
#     return sc_hospitals_points

# def calculate_nearest_hospital(sc_zip_boundary_gdf_with_centroid, sc_hospitals_points):
#     def find_nearest_hospital(centroid):
#         nearest_geom = nearest_points(centroid, gpd.GeoSeries(sc_hospitals_points))[1]
#         return nearest_geom

#     sc_zip_boundary_gdf_with_centroid['nearest_hospital'] = sc_zip_boundary_gdf_with_centroid['centroid'].apply(find_nearest_hospital)
#     return sc_zip_boundary_gdf_with_centroid

# def calculate_distance(sc_zip_boundary_gdf_with_nearest_hospital):
#     sc_zip_boundary_gdf_with_nearest_hospital['distance'] = sc_zip_boundary_gdf_with_nearest_hospital.apply(
#         lambda row: row['centroid'].distance(row['nearest_hospital']), axis=1)
#     return sc_zip_boundary_gdf_with_nearest_hospital

# def create_choropleth_map(sc_zip_boundary_gdf_with_distance):
#     sc_zip_boundary_gdf_with_distance.plot(column='distance', cmap='viridis', legend=True, figsize=(15, 10))
#     plt.title('Distance to Nearest Hospital')
#     plt.show()

# def assembely_solution():
#     sc_zip_boundary_url = "https://github.com/GIBDUSC/test/raw/master/sc_zip_boundary.zip"
#     sc_hospitals_url = "https://github.com/gladcolor/spatial_data/raw/master/South_Carolina/SC_hospitals_with_emergency_room_cleaned.csv"

#     sc_zip_boundary_gdf = load_sc_zip_boundary(sc_zip_boundary_url)
#     sc_hospitals_gdf = load_sc_hospitals(sc_hospitals_url)
    
#     sc_zip_boundary_gdf_with_centroid = calculate_centroid(sc_zip_boundary_gdf)
#     sc_hospitals_points = create_points_from_hospitals(sc_hospitals_gdf)
    
#     sc_zip_boundary_gdf_with_nearest_hospital = calculate_nearest_hospital(sc_zip_boundary_gdf_with_centroid, sc_hospitals_points)
#     sc_zip_boundary_gdf_with_distance = calculate_distance(sc_zip_boundary_gdf_with_nearest_hospital)
    
#     create_choropleth_map(sc_zip_boundary_gdf_with_distance)
#     return sc_zip_boundary_gdf_with_distance

# if __name__ == '__main__':
#     assembely_solution()

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point

def load_sc_zip_boundary(sc_zip_boundary_url):
    """
    Load SC zipcode boundary shapefile

    Parameters:
    sc_zip_boundary_url (str): URL of the SC zipcode boundary shapefile

    Returns:
    gdf (geopandas.GeoDataFrame): SC zipcode boundary GeoDataFrame
    """
    gdf = gpd.read_file(sc_zip_boundary_url)
    return gdf

def load_sc_hospitals(sc_hospitals_url):
    """
    Load SC hospitals data

    Parameters:
    sc_hospitals_url (str): URL of the SC hospitals CSV file

    Returns:
    df (pandas.DataFrame): SC hospitals DataFrame
    """
    df = pd.read_csv(sc_hospitals_url)
    return df

def calculate_centroid(sc_zip_boundary_gdf):
    """
    Calculate centroid for each ZIP code boundary

    Parameters:
    sc_zip_boundary_gdf (geopandas.GeoDataFrame): SC zipcode boundary GeoDataFrame

    Returns:
    gdf (geopandas.GeoDataFrame): SC zipcode boundary with centroid GeoDataFrame
    """
    sc_zip_boundary_gdf['centroid'] = sc_zip_boundary_gdf['geometry'].centroid
    return sc_zip_boundary_gdf

def calculate_distance(sc_zip_boundary_gdf, sc_hospitals_df):
    """
    Calculate distance from each ZIP code centroid to nearest hospital

    Parameters:
    sc_zip_boundary_gdf (geopandas.GeoDataFrame): SC zipcode boundary with centroid GeoDataFrame
    sc_hospitals_df (pandas.DataFrame): SC hospitals DataFrame

    Returns:
    df (pandas.DataFrame): DataFrame with ZIP code and distance to nearest hospital
    """
    distances = []
    for idx, row in sc_zip_boundary_gdf.iterrows():
        centroid = row['centroid']
        nearest_hospital_distance = float('inf')
        for idx2, row2 in sc_hospitals_df.iterrows():
            hospital_location = Point(row2['POINT_X'], row2['POINT_Y'])
            distance = centroid.distance(hospital_location)
            if distance < nearest_hospital_distance:
                nearest_hospital_distance = distance
        distances.append({'ZIPCODE': row['ZCTA5CE10'], 'distance': nearest_hospital_distance})
    df = pd.DataFrame(distances)
    return df

def add_hospital(sc_hospitals_df, sc_zip_boundary_distance_df, sc_zip_boundary_gdf):
    """
    Add hospitals and distance to nearest hospital to map

    Parameters:
    sc_hospitals_df (pandas.DataFrame): SC hospitals DataFrame
    sc_zip_boundary_distance_df (pandas.DataFrame): DataFrame with ZIP code and distance to nearest hospital
    sc_zip_boundary_gdf (geopandas.GeoDataFrame): SC zipcode boundary GeoDataFrame

    Returns:
    str: Path to saved map image
    """
    sc_hospitals_gdf = gpd.GeoDataFrame(sc_hospitals_df, geometry=gpd.points_from_xy(sc_hospitals_df['POINT_X'], sc_hospitals_df['POINT_Y']))
    sc_hospitals_gdf.crs = "EPSG:4326"
    sc_hospitals_gdf = sc_hospitals_gdf.to_crs(epsg=3857)
    
    sc_zip_boundary_gdf = sc_zip_boundary_gdf.merge(sc_zip_boundary_distance_df, left_on='ZCTA5CE10', right_on='ZIPCODE')
    
    fig, ax = plt.subplots(figsize=(15, 10))
    sc_zip_boundary_gdf.plot(column='distance', legend=True, cmap='viridis', ax=ax)
    sc_hospitals_gdf.plot(ax=ax, color='red', markersize=10, label='Hospitals')
    
    ax.set_title('Distance to the Nearest Hospital')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.legend()
    
    plt.savefig('final_map.png')
    return "final_map.png"

def assembely_solution():
    sc_zip_boundary_url = 'https://github.com/GIBDUSC/test/raw/master/sc_zip_boundary.zip'
    sc_hospitals_url = "https://github.com/gladcolor/spatial_data/raw/master/South_Carolina/SC_hospitals_with_emergency_room_cleaned.csv"

    sc_zip_boundary_gdf = load_sc_zip_boundary(sc_zip_boundary_url)
    sc_hospitals_df = load_sc_hospitals(sc_hospitals_url)
    
    sc_zip_boundary_centroid_gdf = calculate_centroid(sc_zip_boundary_gdf)
    sc_zip_boundary_distance_df = calculate_distance(sc_zip_boundary_centroid_gdf, sc_hospitals_df)
    
    final_map_path = add_hospital(sc_hospitals_df, sc_zip_boundary_distance_df, sc_zip_boundary_gdf)
    
    return final_map_path

if __name__ == '__main__':
    final_map_path = assembely_solution()
    print(f"Final map saved to: {final_map_path}")
