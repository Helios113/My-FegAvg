# MM-FedAvg

MM-FedAvg is a compilation of federated algorithms which I developed for my master's thesis. These algorithms enable multimodal FL to run faster and be more robust.

## Project components
This project contains 5 major components:
- Server
- Client
- MHealthDataset
- Models
- Data handler

This code can be adapted to any data set. You only need to change MHealthDataset and implement the models you need. 

## Server
Server script mimics the function of a real server. It creates and initializes any number of devices. The initialization process requires one to provide names for each modality which are later used to perform parameter averaging (local and/or global).

## Client
Client script mimics the function of a client device. It stores the model data and performs training. 
## MHealthDataset
The MHealthDataset class reads the MHealth dataset and loads it into a pytorch understandable format.
## Models
Models script stores different models used by the client devices to train models.
## Data handler
Data_handler script conditions the raw data from MHealth and saves it for loading into the MHealthDataset class.