# Vehicles-Dispatch-Simulator
This simulator serves as the training and evaluation platform in the following work:  
[Context-Aware Taxi Dispatching at City-Scale Using Deep Reinforcement Learning](http://www.com)  

## Introduction
This simulator is used to simulate urban vehicle traffic.  
The system divides the day into several time slots. System information is updated at the beginning of each time slot to update vehicle arrivals and order completion. Then the system generates the order that needs to be started within the current time slice, and then finds the optimal idle vehicle to match the order. If the match fails or the recent vehicles have timed out, the order is marked as Reject. If it is successful, the vehicle service order is arranged. The shortest path in the road network first reaches the place where the order occurred, and then arrives at the order destination, and repeats matching the order until all the orders in the current time slice have been completed. Then the system generates orders that occur within the current time slice, finds the nearest idle vehicle to match the order, and if there is no idle vehicle or the nearest idle vehicle reaches the current position of the order and exceeds the limit time, the match fails, and if the match is successful, the selected vehicle service is arranged Order. After the match is successful, the vehicle's idle record in the current cluster is deleted, and the time to be reached is added to the cluster where the order destination is located. The vehicle must first arrive at the place where the order occurred, pick up the passengers, and then complete the order at the order destination. Repeat the matching order until a match All orders in this phase are completed.  
At the end of the matching phase, you can use your own matching method to dispatch idle vehicles in each cluster to other clusters that require more vehicles to meet future order requirements.


## Data source
- The order data used by the simulator comes from [Didi Chuxing](https://gaia.didichuxing.com)
- The map data used by the simulator comes from [OpenStreetMap](https://www.openstreetmap.org)
- The weather data used by the simulator comes from [China Meteorological Administration](http://www.cma.gov.cn/)

## Prerequisites
- **Python 3**
- **NumPy**
- **Pandas**
- **Matplotlib**

## Run
#### Run a simple simulation program:
    
    cd ./Vehicles-Dispatch-Simulator/
    python Demo_simulation.py    
If you want to customize more functions, please enter the settings.py under the config folder to modify the parameters of the simulator. You can even adjust the size of the experimental area by entering latitude and longitude. The order, road network and grid division will be adaptive. Adjust to fit selected area

## Architecture
#### Cluster / Grid
The whole city is divided into several clusters or Grids, each cluster includes its own unique ID and several road intersections, and each cluster has an idle vehicle table to record the idle vehicles in the current time slot. In addition, each cluster has a vehicle arrival table to record vehicles that will arrive in the future.  
<br>
- An example of dividing urban areas by clusters  
![](https://github.com/szlhl1040/Simulator/blob/master/CARnet%20clustering.png)

#### Vehicles
Each vehicle is randomly assigned to any cluster. Each vehicle has a record of its current location and future destination. There is also an order set. When a vehicle transports passengers, the order is loaded into the order set of the vehicle. When Remove this order when it reaches its destination.

#### Orders
The order is derived from [Didi Chuxing](https://gaia.didichuxing.com). You can download the complete dataset on its official website. Here, only one day of order data is provided as a test case.Each day contains more than 200,000 order records, each order records the time of occurrence, location of occurrence, and destination location, where the location of the order is bound to the nearest road intersection

#### Match module
Each matching module will match the orders that will occur within the current time slot. The matching module will find the nearest idle vehicles for each order. It can also enable the neighbor car search system to determine the search range according to the set search distance and the size of the grid. It use dfs to find the nearest idle vehicles in the area.

#### Prediction module
We provide weather data corresponding to the order time, which can better serve the order distribution and quantity forecast. You can implement your own order forecasting method to provide efficient and accurate help for Dispatch method

#### Dispatch module
you can implement your own Dispatch method in Dispatch module to move idle vehicles in each cluster to other clusters

