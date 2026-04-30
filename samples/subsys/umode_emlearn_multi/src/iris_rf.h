


    // !!! This file is generated using emlearn !!!

    #include <stdint.h>
    

static inline int32_t iris_rf_tree_0(const float *features, int32_t features_length) {
          if (features[3] < 0.750000f) {
              return 0;
          } else {
              if (features[2] < 4.850000f) {
                  if (features[3] < 1.650000f) {
                      return 1;
                  } else {
                      if (features[1] < 3.000000f) {
                          return 2;
                      } else {
                          return 1;
                      }
                  }
              } else {
                  if (features[0] < 6.600000f) {
                      return 2;
                  } else {
                      if (features[2] < 5.200000f) {
                          return 1;
                      } else {
                          return 2;
                      }
                  }
              }
          }
        }
        

static inline int32_t iris_rf_tree_1(const float *features, int32_t features_length) {
          if (features[3] < 0.800000f) {
              return 0;
          } else {
              if (features[3] < 1.750000f) {
                  if (features[2] < 4.950000f) {
                      return 1;
                  } else {
                      if (features[2] < 5.450000f) {
                          return 1;
                      } else {
                          return 2;
                      }
                  }
              } else {
                  if (features[3] < 1.900000f) {
                      if (features[2] < 4.850000f) {
                          return 2;
                      } else {
                          return 2;
                      }
                  } else {
                      return 2;
                  }
              }
          }
        }
        

static inline int32_t iris_rf_tree_2(const float *features, int32_t features_length) {
          if (features[0] < 5.550000f) {
              if (features[3] < 0.800000f) {
                  return 0;
              } else {
                  if (features[3] < 1.600000f) {
                      return 1;
                  } else {
                      return 2;
                  }
              }
          } else {
              if (features[3] < 1.550000f) {
                  if (features[3] < 0.750000f) {
                      return 0;
                  } else {
                      if (features[2] < 5.000000f) {
                          return 1;
                      } else {
                          return 2;
                      }
                  }
              } else {
                  if (features[2] < 4.650000f) {
                      return 1;
                  } else {
                      if (features[3] < 1.700000f) {
                          return 2;
                      } else {
                          return 2;
                      }
                  }
              }
          }
        }
        

static inline int32_t iris_rf_tree_3(const float *features, int32_t features_length) {
          if (features[0] < 5.450000f) {
              if (features[1] < 2.800000f) {
                  if (features[1] < 2.450000f) {
                      return 1;
                  } else {
                      if (features[0] < 5.000000f) {
                          return 2;
                      } else {
                          return 1;
                      }
                  }
              } else {
                  return 0;
              }
          } else {
              if (features[0] < 6.250000f) {
                  if (features[3] < 1.700000f) {
                      if (features[3] < 0.600000f) {
                          return 0;
                      } else {
                          return 1;
                      }
                  } else {
                      return 2;
                  }
              } else {
                  if (features[3] < 1.650000f) {
                      return 1;
                  } else {
                      return 2;
                  }
              }
          }
        }
        

static inline int32_t iris_rf_tree_4(const float *features, int32_t features_length) {
          if (features[3] < 0.700000f) {
              return 0;
          } else {
              if (features[3] < 1.750000f) {
                  if (features[2] < 5.050000f) {
                      if (features[2] < 4.950000f) {
                          return 1;
                      } else {
                          return 1;
                      }
                  } else {
                      if (features[3] < 1.550000f) {
                          return 2;
                      } else {
                          return 1;
                      }
                  }
              } else {
                  return 2;
              }
          }
        }
        

static inline int32_t iris_rf_tree_5(const float *features, int32_t features_length) {
          if (features[3] < 0.800000f) {
              return 0;
          } else {
              if (features[2] < 4.950000f) {
                  if (features[0] < 4.950000f) {
                      if (features[3] < 1.350000f) {
                          return 1;
                      } else {
                          return 2;
                      }
                  } else {
                      if (features[2] < 4.750000f) {
                          return 1;
                      } else {
                          return 1;
                      }
                  }
              } else {
                  return 2;
              }
          }
        }
        

static inline int32_t iris_rf_tree_6(const float *features, int32_t features_length) {
          if (features[3] < 0.700000f) {
              return 0;
          } else {
              if (features[2] < 4.750000f) {
                  if (features[0] < 4.950000f) {
                      return 2;
                  } else {
                      return 1;
                  }
              } else {
                  if (features[2] < 5.150000f) {
                      if (features[0] < 6.600000f) {
                          return 2;
                      } else {
                          return 1;
                      }
                  } else {
                      return 2;
                  }
              }
          }
        }
        

static inline int32_t iris_rf_tree_7(const float *features, int32_t features_length) {
          if (features[2] < 2.600000f) {
              return 0;
          } else {
              if (features[2] < 4.750000f) {
                  return 1;
              } else {
                  if (features[2] < 5.150000f) {
                      if (features[3] < 1.750000f) {
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
        

static inline int32_t iris_rf_tree_8(const float *features, int32_t features_length) {
          if (features[3] < 0.700000f) {
              return 0;
          } else {
              if (features[0] < 6.250000f) {
                  if (features[2] < 4.800000f) {
                      if (features[0] < 4.950000f) {
                          return 2;
                      } else {
                          return 1;
                      }
                  } else {
                      if (features[1] < 2.650000f) {
                          return 2;
                      } else {
                          return 2;
                      }
                  }
              } else {
                  if (features[2] < 5.050000f) {
                      if (features[0] < 6.350000f) {
                          return 1;
                      } else {
                          return 1;
                      }
                  } else {
                      return 2;
                  }
              }
          }
        }
        

static inline int32_t iris_rf_tree_9(const float *features, int32_t features_length) {
          if (features[2] < 2.600000f) {
              return 0;
          } else {
              if (features[2] < 4.950000f) {
                  if (features[0] < 5.950000f) {
                      return 1;
                  } else {
                      if (features[3] < 1.650000f) {
                          return 1;
                      } else {
                          return 2;
                      }
                  }
              } else {
                  if (features[0] < 6.600000f) {
                      return 2;
                  } else {
                      if (features[0] < 6.750000f) {
                          return 2;
                      } else {
                          return 2;
                      }
                  }
              }
          }
        }
        

int32_t iris_rf_predict(const float *features, int32_t features_length) {

        int32_t votes[3] = {0,};
        int32_t _class = -1;

        _class = iris_rf_tree_0(features, features_length); votes[_class] += 1;
    _class = iris_rf_tree_1(features, features_length); votes[_class] += 1;
    _class = iris_rf_tree_2(features, features_length); votes[_class] += 1;
    _class = iris_rf_tree_3(features, features_length); votes[_class] += 1;
    _class = iris_rf_tree_4(features, features_length); votes[_class] += 1;
    _class = iris_rf_tree_5(features, features_length); votes[_class] += 1;
    _class = iris_rf_tree_6(features, features_length); votes[_class] += 1;
    _class = iris_rf_tree_7(features, features_length); votes[_class] += 1;
    _class = iris_rf_tree_8(features, features_length); votes[_class] += 1;
    _class = iris_rf_tree_9(features, features_length); votes[_class] += 1;
    
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
    

int iris_rf_predict_proba(const float *features, int32_t features_length, float *out, int out_length) {

        int32_t _class = -1;

        for (int i=0; i<out_length; i++) {
            out[i] = 0.0f;
        }

        _class = iris_rf_tree_0(features, features_length); out[_class] += 1.0f;
    _class = iris_rf_tree_1(features, features_length); out[_class] += 1.0f;
    _class = iris_rf_tree_2(features, features_length); out[_class] += 1.0f;
    _class = iris_rf_tree_3(features, features_length); out[_class] += 1.0f;
    _class = iris_rf_tree_4(features, features_length); out[_class] += 1.0f;
    _class = iris_rf_tree_5(features, features_length); out[_class] += 1.0f;
    _class = iris_rf_tree_6(features, features_length); out[_class] += 1.0f;
    _class = iris_rf_tree_7(features, features_length); out[_class] += 1.0f;
    _class = iris_rf_tree_8(features, features_length); out[_class] += 1.0f;
    _class = iris_rf_tree_9(features, features_length); out[_class] += 1.0f;
    
        // compute mean
        for (int i=0; i<out_length; i++) {
            out[i] = out[i] / 10;
        }
        return 0;
    }
    