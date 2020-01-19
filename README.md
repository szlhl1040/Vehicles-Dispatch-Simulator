# Vehicles-Dispatch-Simulator
This simulator serves as the training and evaluation platform in the following work:  
[Context-Aware Taxi Dispatching at City-Scale Using Deep Reinforcement Learning](http://www.com)  

## Introduction
This simulator is used to simulate urban vehicle traffic.  

Each road intersection is used as a vertex, the edges between the road intersections represent roads, and the weights of the edges represent the distance cost of this road.
## Data source
- The order data used by the simulator comes from [Didi Chuxing](https://gaia.didichuxing.com)
- The map data used by the simulator comes from [OpenStreetMap](https://www.openstreetmap.org)

## ku
- **Python 3**
- **NumPy**
- **Pandas**
- **Matplotlib**

## Architecture
#### Cluster / Grid
The whole city is divided into several clusters or Grids, each cluster includes its own unique ID and several road intersections, and each cluster has an idle vehicle table to record the idle vehicles in the current time slot. In addition, each cluster has a vehicle arrival table to record vehicles that will arrive in the future.
#### Vehicles
Each vehicle is randomly assigned to any cluster. Each vehicle has a record of its current location and future destination. There is also an order set. When a vehicle transports passengers, the order is loaded into the order set of the vehicle. When Remove this order when it reaches its destination.
#### 

