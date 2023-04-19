# Experiements

1. UniFed vs single devices
    - acc data

    1. Find out how many epochs and how many rounds of training to perform
        How? Maybe I can first run fedAvg, no-fed, global for 500 and 1000 and check the diff in loss
            For both: Adam with 500 total iteration gives the best results

        So now I need to try unimodal vs glob_fed will run on Adam

    How I need to save loss and F1 score


What is important is to check if the vectors are the same after training.
We need to compare the 4 different modalities on the same device. Unimodal vs UniMFed. Then look at the feature vector; iid and non iid
We need to compare multi m fed vs 4 clinents with early fusion iid and non-iid
We need to compare unim-fed vs 4 modalities on seperate devices - prob will such, but still look at the feature vectors
