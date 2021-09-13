Execute the following steps to recreate figures 5 and C1 with saliency_maps.py

1. Produce 4 light-cones in `../simulations` with the same predefined parameters
    ```
    python3 runSimulations.py --saliency [options] 4
    ```
2. Create opt mocks in `../mock_creation` with
    ```
    python3 create_mocks.py --saliency
    ```
3. Run 
    ```
    python3 saliency_maps.py
    ```
