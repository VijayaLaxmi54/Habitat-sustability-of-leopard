import pandas as pd
import numpy as np
import os

def generate_synthetic_data(num_samples=1000, output_path='dataset.csv'):
    np.random.seed(42)
    
    # Generate features
    # Suitable paths (target=1): High NDVI, moderate/high elevation, moderate slope, far from roads/settlements, close to water, favorable land use (forests=1, scrub=2)
    # Unsuitable (target=0): Low NDVI, either very high/low elevation, very steep/flat slope, close to roads/settlements, far from water, unfavorable land use (urban=3, agriculture=4)
    
    targets = np.random.choice([0, 1], size=num_samples, p=[0.6, 0.4])
    
    data = []
    for t in targets:
        if t == 1:
            ndvi = np.random.normal(0.7, 0.1)
            elevation = np.random.normal(1200, 200)
            slope = np.random.normal(15, 5)
            land_use = np.random.choice([1, 2], p=[0.7, 0.3]) # 1: Forest, 2: Scrub
            dist_water = np.random.normal(500, 200) # close to water (meters)
            dist_roads = np.random.normal(5000, 1000) # far from roads (meters)
            dist_settlements = np.random.normal(6000, 1500) # far from settlements
        else:
            ndvi = np.random.normal(0.3, 0.15)
            elevation = np.random.normal(800, 300)
            slope = np.random.normal(5, 5)
            land_use = np.random.choice([3, 4], p=[0.4, 0.6]) # 1: Forest, 2: Scrub, 3: Urban, 4: Agriculture
            dist_water = np.random.normal(2000, 800) # far from water
            dist_roads = np.random.normal(1000, 500) # close to roads
            dist_settlements = np.random.normal(1500, 800) # close to settlements
            
        data.append([
            max(0, min(1, ndvi)), max(0, elevation), max(0, slope), land_use, 
            max(0, dist_water), max(0, dist_roads), max(0, dist_settlements), t
        ])
        
    df = pd.DataFrame(data, columns=[
        'NDVI', 'Elevation', 'Slope', 'Land_Use', 
        'Distance_to_Water', 'Distance_to_Roads', 'Distance_to_Settlements', 'Habitat_Suitability'
    ])
    
    df.to_csv(output_path, index=False)
    print(f"Dataset generated at {output_path} with {num_samples} samples.")

if __name__ == '__main__':
    generate_synthetic_data()
