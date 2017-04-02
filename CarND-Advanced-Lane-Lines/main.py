from calibration import load_calib_data



def main():

    # Load camera matrix and distortion parameters from pickle file
    calib_mtx, calib_dist = load_calib_data("calib_data.p")
    print ("Calibration Matrix and Distance")
    print ("-------------------------------")
    print (calib_mtx)
    print (calib_dist)

if __name__ == '__main__':
    main()
