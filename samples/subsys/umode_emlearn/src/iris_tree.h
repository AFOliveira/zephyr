


    // !!! This file is generated using emlearn !!!

    #include <stdint.h>


static inline int32_t iris_tree_tree_0(const float *features, int32_t features_length) {
          if (features[3] < 0.800000f) {
              return 0;
          } else {
              if (features[3] < 1.750000f) {
                  if (features[2] < 4.950000f) {
                      if (features[3] < 1.650000f) {
                          return 1;
                      } else {
                          return 2;
                      }
                  } else {
                      if (features[3] < 1.550000f) {
                          return 2;
                      } else {
                          return 1;
                      }
                  }
              } else {
                  if (features[2] < 4.850000f) {
                      if (features[0] < 5.950000f) {
                          return 1;
                      } else {
                          return 2;
                      }
                  } else {
                      return 2;
                  }
              }
          }
        }


int32_t iris_tree_predict(const float *features, int32_t features_length) {

        int32_t votes[3] = {0,};
        int32_t _class = -1;

        _class = iris_tree_tree_0(features, features_length); votes[_class] += 1;

        int32_t most_voted_class = -1;
        int32_t most_voted_votes = 0;
        for (int32_t i=0; i<3; i++) {

            if (votes[i] > most_voted_votes) {
                most_voted_class = i;
                most_voted_votes = votes[i];
            }
        }
        return most_voted_class;
    }


int iris_tree_predict_proba(const float *features, int32_t features_length, float *out, int out_length) {

        int32_t _class = -1;

        for (int i=0; i<out_length; i++) {
            out[i] = 0.0f;
        }

        _class = iris_tree_tree_0(features, features_length); out[_class] += 1.0f;

        // compute mean
        for (int i=0; i<out_length; i++) {
            out[i] = out[i] / 1;
        }
        return 0;
    }
