#include <stdio.h>
#include "nnet.h"

int main() {
    NNet* nnet = load_network("./nnet/model_mlp.nnet");
    if (nnet == NULL) {
        printf("Failed to load network\n");
        return 1;
    }
    
    printf("Network loaded successfully!\n");
    printf("  Input size: %d\n", nnet->inputSize);
    printf("  Output size: %d\n", nnet->outputSize);
    printf("  Number of layers: %d\n", nnet->numLayers);
    printf("  Max layer size: %d\n", nnet->maxLayerSize);
    
    // Print layer sizes
    printf("  Layer sizes: ");
    for (int i = 0; i <= nnet->numLayers; i++) {
        printf("%d ", nnet->layerSizes[i]);
    }
    printf("\n");
    
    destroy_network(nnet);
    return 0;
}
