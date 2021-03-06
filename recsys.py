## Design Exchange
## Recommendation System
##
## Created by:  Adam Spitzig
## Created on:  Nov 24, 2014

# Dependencies:
# numpy, scipy, nimfa, matplotlib, mpltools, sparsesvd, requests

# Temp fix to ensure nimfa is importable
import sys
sys.path.append('/usr/local/lib/python2.7/dist-packages/nimfa-1.0-py2.7.egg')

# Imports
import numpy as np
import pylab as pl
import pandas as pd
from sklearn.pls import PLSCanonical, PLSRegression, CCA  #<-- Modify depending on specific use
import scipy.stats as stats
import scipy.spatial.distance as dist
from scipy.linalg import svd,diagsvd
from scipy import sparse
import nimfa
import time
import matplotlib.pyplot as plt
from mpltools import style
from sparsesvd import sparsesvd
from mpltools import style
style.use('ggplot')
import sqlite3 as lite
import requests
import json


# LOCAL DB INFO

# SQLite database path
SQLITE_DB = r"/home/aspitzig/Desktop/DesEx/design_exchange_development.sqlite3"

# Cases, methods and cases-to-methods table names and field names
##ALL_METHOD_TBL = "design_methods"
##ALL_METHOD_FLD = "name"
##ALL_CASE_TBL = "case_studies"
##ALL_CASE_FLD = "title"
ALL_ID_FLD = "id"                        # for both all_case_tbl and all_method_tbl (code presums same field name in both)
##C2M_DB_TBL = "method_case_studies"
##C2M_CASE_FLD = "case_study_id"
##C2M_METHOD_FLD = "design_method_id"


# API INFO
ROOT_API_URL = "https://thedesignexchange.herokuapp.com/"           # root API url for Design Exchange web app
API_ALL_METHOD_TBL_EXT = "design_methods"                           # all design methods extension
API_ALL_METHOD_TBL_BY_ID_EXT = "design_methods/id"                  # design methods by ID extension
API_ALL_CASE_TBL_EXT = "case_studies"                               # all case studies extension
API_ALL_CASE_TBL_BY_ID_EXT = "case_studies/id"                      # case studies by ID extension
API_C2M_TBL_EXT = "method_case_studies"                             # methods/case studies (one-to-many tbl) extension
API_ALL_CASE_FLD = "name"
API_ALL_METHOD_FLD = "name"
API_C2M_CASE_FLD = "case_study_id"
API_C2M_METHOD_FLD = "design_method_id"


# CASE ATTRIBUTE INFO (from "case_studies" table)
CASE_ATTRIB_DICT = {"development_cycle":2,
                    "design_phase":2,
                    "project_domain":4,
                    "customer_type":4,
                    "user_age":3,
                    "privacy_level":1,
                    "social_setting":2,
                    "customerIsUser":"boolean",
                    "remoteProject":"boolean"}
# Dict structure: KEY -> attribute field name (string) | VALUE-> max category value (int) OR indication that attrib is boolean (string)
# Note that max category values represents highest of INCLUSIVE range - i.e. 3 means range (0, 3) inclusive [0, 1, 2 or 3]
# Note also that, in local database, boolean attribute values are "f" or "t"

# Note that "attributes" here refers to combination of case "characteristics" and "details" (terminology on website)


# GET DATA FROM LOCAL DATABASE
# Using hard-coded db filepaths defined above

def get_all_cases_and_methods_data(db, case_tbl, case_name_fld, method_tbl, method_name_fld, id_fld="id"):
    ''' 
    input:  str, full path to target sqlite database (db)
            str, name of target database table with case studies (case_tbl)
            str, name of field with case names (case_name_fld)
            str, name of target database table with methods (method_tbl)
            str, name of field with method names (method_name_fld)
            str, name of id field for all (id_fld) (presumed to be "id", which is default)
    output: two arrays, one for methods and one for cases, each w/ two cols: id, name
    '''
        
    con = None
    try:
        con = lite.connect(db)    
        cur = con.cursor()
        SQL_cases = "SELECT " + id_fld + ", " + case_name_fld + " FROM " + case_tbl
        SQL_methods = "SELECT " + id_fld + ", " + method_name_fld + " FROM " + method_tbl 
        cur.execute(SQL_cases)
        result_cases = np.array(cur.fetchall())
        cur.execute(SQL_methods)
        result_methods = np.array(cur.fetchall())
        return result_cases, result_methods
    except lite.Error, e:    
        print "Error %s:" % e.args[0]
        sys.exit(1)    
    finally:
        if con:
            con.close()

def get_cases_to_methods_data(db, tbl, case_fld, method_fld):
    ''' 
    input:  str, full path to target sqlite database (db)
            str, name of target database table (method-case study relation) (tbl)
            str, name of field with cases in target table (case_fld)
            str, name of field with methods in target table (method_fld)
    output: data in cases and methods fields in table, as array

    NOTE:   consider consolidating this function with get_all_cases_and_methods_data() above, in one get_data function
            OR both with create_interaction_matrix() function, in one big create_interaction_matrix function (which gets data and creates matrix)
    '''

    con = None
    try:
        con = lite.connect(db)    
        cur = con.cursor()
        SQL = "SELECT " + case_fld + ", " + method_fld + " FROM " + tbl
        cur.execute(SQL)
        result = np.array(cur.fetchall())
        return result
    except lite.Error, e:    
        print "Error %s:" % e.args[0]
        sys.exit(1)    
    finally:
        if con:
            con.close()



# GET DATA VIA API
# Using hard-coded API info above

def get_all_data_via_api(url, case_tbl_ext, case_tbl_name_fld, method_tbl_ext, method_tbl_name_fld, c2m_tbl_ext, c2m_case_fld, c2m_method_fld, id_fld):

    ''' 
    input:  str, root DesEx url (url)
            str, case table extension (name of case table, to append to root url) (case_tbl_ext)
            str, name of field with case names (case_tbl_name_fld)
            str, method table extension (name of method table, to append to root url) (method_tbl_ext)
            str, name of field with method names (method_tbl_name_fld)
            str, case-to-method table extension (name of case-to-method table, to append to root url) (c2m_tbl_ext)
            str, name of field with case ids in c-to-m table (c2m_case_fld)
            str, name of field with method ids in c-to-m table (c2m_method_fld)
    output: three arrays:
                (1) all methods, w/ two cols: id, name
                (2) all cases, w/ two cols: id, name
                (3) case-to-methods, w/ two cols: case_id, method_id

    dependencies: requests, json
    '''
    
    # Construct request urls
    case_tbl_url = url + case_tbl_ext + ".json"
    method_tbl_url = url + method_tbl_ext + ".json"
    c2m_tbl_url = url + c2m_tbl_ext + ".json"

    # TEST
    #print "case_tbl_url"
    #print case_tbl_url
    #print "method_tbl_url"
    #print method_tbl_url
    #print "c2m_tbl_url"
    #print c2m_tbl_url
    #print
    #print "getting data via API..."

    # Get data via urls (as pandas dfs)
    case_tbl_data = pd.io.json.read_json(case_tbl_url)
    method_tbl_data = pd.io.json.read_json(method_tbl_url)
    c2m_tbl_data = pd.io.json.read_json(c2m_tbl_url)

    # Construct column lists (i.e columns to keep for output arrays)
    case_cols = [id_fld, case_tbl_name_fld]
    method_cols = [id_fld, method_tbl_name_fld]
    c2m_cols = [c2m_case_fld, c2m_method_fld]

    # TEST
    #print case_cols
    #print method_cols
    #print c2m_cols
    #print case_tbl_data

    # Sort pandas dfs (using column lists from above as: [sort by this col first, sort by this col second])
    case_tbl_data_sort = case_tbl_data.sort(columns=[case_cols[0]])
    method_tbl_data_sort = method_tbl_data.sort(columns=[method_cols[0]])
    c2m_tbl_data_sort = c2m_tbl_data.sort(columns=[c2m_cols[0]])

    # Convert pandas dfs to numpy arrays, removing all but columns to keep   
    case_tbl_final = case_tbl_data_sort.as_matrix(columns = case_cols)
    method_tbl_final = method_tbl_data_sort.as_matrix(columns = method_cols)
    c2m_tbl_final = c2m_tbl_data_sort.as_matrix(columns = c2m_cols)

    # TEST
    #print "case_tbl_data_sort_clean"
    #print case_tbl_data_sort_clean
    #print "case_tbl_data"
    #print case_tbl_data
    
    #print "method_tbl_data_sort_clean[:5,]"
    #print method_tbl_data_sort_clean[:5,]
    #print "method_tbl_data"
    #print method_tbl_data
    
    #print "c2m_tbl_data_sort_clean[:5,]"
    #print c2m_tbl_data_sort_clean[:5,]
    #print "c2m_tbl_data"
    #print "TYPE"
    #print type(c2m_tbl_data)
    #print c2m_tbl_data


    return case_tbl_final, method_tbl_final, c2m_tbl_final


def create_interaction_matrix(all_cases, all_methods, cases_to_methods, id_fld):
    '''
    input:  array, all data in cases table (id and name) (product 1 of get_all_data_via_api() function
                and also get_all_cases_and_methods_data() function) (all_cases)
            array, all data in methods table (id and name) (product 2 of get_all_data_via_api() function
                and also get_all_cases_and_methods_data() function) (all_methods)
            array, data in cases-to-methods relatioship table (product 3 of get_all_data_via_api() function
                and also sole product of get_cases_to_methods() function) (cases_to_methods)
    output: array, matrix representing the interaction of cases and methods (binary interaction matrix)

    note: all_cases and all_methods arrays must be ordered by id (ascending) and first column of both must be id field
    '''

    # create list of all case and method ids (as integers)
    case_ids = list(all_cases[:,0]) 
    case_ids = map(int, case_ids)
    method_ids = list(all_methods[:,0])
    method_ids = map(int, method_ids)

    # Create empty matrix (all zeros) of size cases*methods
    cases_len = len(all_cases)
    methods_len = len(all_methods)    
    cases_methods_matrix = np.zeros((cases_len, methods_len))
    cases_methods_matrix_alt = np.zeros((cases_len, methods_len))

    # Fill matrix (with "1"s) according to cases_to_methods relationship table
    for case_id, method_id in cases_to_methods:
        # TEST
        print "Case id: " + str(case_id) + "  Method id: " + str(method_id)

        # original construction of matrix, reliant on unbroken id sequence (works for unbroken/ordered id seq inputs)
        #cases_methods_matrix[int(case_id)-1][int(method_id)-1] = 1
        # note: all matrix row numbers correspond to (case_id - 1) and column numbers to (method_id - 1) 

        # alternative construction, unreliant on unbroken id sequence (appears to work, after initial checks)
        case_id_index = case_ids.index(case_id)
        method_id_index = method_ids.index(method_id)
        cases_methods_matrix_alt[case_id_index][method_id_index] = 1

    # TEST
    #print "MATRICES EQUIVALENT: "
    #print np.array_equal(cases_methods_matrix, cases_methods_matrix_alt)    

    return cases_methods_matrix_alt  
    
def case_to_case_collab_filter(int_matrix, k=3, thresh=0.5):
    '''
    input:  array, interaction matrix (int_matrix)
            int, number of nearest neighbors - i.e. number of 'closest' cases to be used to determine prediced values (k) 
            float, cutoff probability above which a method is considered to be probable/recommendable - i.e. marked as '1' in output
    output: array, binary matrix w/ predicted values (0 or 1)   

    NOTE:   THIS CALCULATES JACCARD DISTANCE FOR ALL ROWS IN int_matrix -> POSSIBLE OVERKILL. For DesEx only need to calc distances between
            user-entered case (profile as vector, i.e. single row) and all other cases, and make predictions/recommendations for the single
            input case. Considering modifying in refinement stage.
            
            Also in refinement stage, consider optimizing k and thresh values (using cross-validation, or similar, approach)
    '''
    #create distance matrix
    dist_matrix = dist.squareform(dist.pdist(int_matrix, 'jaccard'))

    #find the closest K cases to all cases in int_matrix
    closest_matrix = np.argsort(dist_matrix)[:,:k]

    #average the closest K cases
    result = np.mean(int_matrix[closest_matrix],1)
    
    #with threshold average of 0.5, convert all to zeros and ones
    result = stats.threshold(result,threshmax=thresh,newval=1)
    result = stats.threshold(result,threshmin=(thresh-0.001),newval=0)

    return result

def method_to_method_collab_filter(int_matrix, k=3, thresh=0.5):
    '''
    input:  array, interaction matrix (int_matrix)
            int, number of nearest neighbors - i.e. number of 'closest' methods to be used to determine prediced values (k) 
            float, cutoff probability above which a case is considered to be probable/recommendable - i.e. marked as '1' in output (thresh)
    output: array, binary matrix w/ predicted values (0 or 1) 

    NOTE:   In refinement stage, consider optimizing k and thresh values (using cross-validation, or similar, approach)

            Some resources indicate that item-to-item (here, method-to-method) collab filtering - this function - may work best
    '''
    #transpose int_matrix (rows are now methods)
    int_matrix_T = int_matrix.T

    #create distance matrix
    dist_matrix = dist.squareform(dist.pdist(int_matrix_T, 'jaccard'))

    #find the closest K methods to all methods
    closest_matrix = np.argsort(dist_matrix)[:,:k]

    #average the closest K methods
    result = np.mean(int_matrix_T[closest_matrix],1)
    
    #with threshold average of 0.5, convert all to zeros and ones
    result = stats.threshold(result,threshmax=thresh,newval=1)
    result = stats.threshold(result,threshmin=(thresh-0.001),newval=0)

    #re-transpose (methods are back to columns)
    final_result = result.T

    return final_result
    

def case_attrib_data_to_binary_array(db, case_tbl, attrib_dict):
    '''
    input:  str, full path to target sqlite database (db)
            str, name of target database table with case studies (case_tbl)
            dict, name of case attribute dictionary w/ those attributes being use in rec system (attrib_dict)
    output:

    Function retrieves case attribute data from database, and converts it to binary matrix, with all attribs as 1/0 dummy vars
    Output ready for use as X-matrix in PLS regression (if using)
    Output also ready for appending interaction matrix, and creating big array ready for use in collab filters (if bypassing PLS regression)

    NOTE: for API purposes, consider inserting this function into the single get_data_via_api function, and including bin matrix as another output
    '''
    import pandas as pd   #pandas used here b/c it more easily allows for conversion to dummy (binary) variables for all attribute cols

    # create string of all desired field names (case attributes) to extract from case table, and list of all needing dummy variable creation
    dummy_fld_list = []
    bool_fld_list = []
    fld_str = "id"                          # start with id field
    for key in attrib_dict.keys():          # add all fields in CASE_ATTRIB_DICT
        fld_str += ", "
        fld_str += key
        if attrib_dict[key] == "boolean":   # add all fields to bool_fld_list that are in boolean form
            bool_fld_list.append(key)
        else:
            dummy_fld_list.append(key)      # add all fields to dummy_fld_list that are polytomous

    # get data from case table in sqlite database AS pandas dataframe
    con = lite.connect(db)
    SQL = "SELECT " + fld_str + " FROM " + case_tbl
    df = pd.read_sql(SQL, con)

    # print df #TEST

    # Consider using the function below to clean the dataset
    # Currently this function is unused and unfinished
    # Best and CURRENT WORKING ASSUMPTION is that all rows (cases) in database have all attributes filled out
    def clean_df(df, min_non_NAs):
        '''
        input: pandas dataframe (df), minimum number of non-NaN values needed to keep a row (otherwise it is deleted) (int)
        output: pandas dataframe, cleaned

        NOTE: to eliminate cases with ANY NaN values, replace "thresh" argument with "how" and set to "any" (as str))
              to eliminate cases with ALL NaN values, " "  " "  "all"
        '''
        return df.dropna(axis=0, thresh=min_non_NAs)

    # add new indicator field for all polytomous variables
    for fld in dummy_fld_list:
        for elem in df[fld].unique():
            df[str(fld) + "_" + str(elem)] = df[fld] == elem

    # remove original polytomous var fields
    for fld in dummy_fld_list:
        df = df.drop(fld, 1)

    ## NOTE! Here, may want to delete newly-added indicator fields that end in "nan" too (b/c original polytomous var field had NaN values) ##
    ##  The comparison above appears not to work with "NaN" in any case, rendering all cells under "nan"-ending cols False
    ## (another reason to allow ONLY cases with all attributes filled out into the RecSys)

    # NOTE new indicator fields are truly boolean, while fields in bool_fld_list from original table are "t" and "f". Therefore...
    # ...change original bool_fld_lst fields from "t"/"f" to True/False
    def all_to_Bool(x):
        if type(x) == unicode:
            str_x = str(x)
            if str_x == "t":
                return True
            elif str_x == "f":
                return False
        else:
            return x
         
    df = df.applymap(all_to_Bool)        

    # convert all values to 0s and 1s
    df = df.astype(int)

    # remove id column
    df = df.drop("id", 1)

    # convert pandas dataframe to numpy array, creating final binary matrix
    case_attrib_bin_matrix = df.as_matrix(columns=None)

    # create list of field names of final binary X matrix, in order
    case_attrib_bin_matrix_fields = [str(x) for x in list(df.columns.values)]
      # NOTE: potentially useful, beyond check, in future refinement/modeling, where indiv case attributes are used, perhaps, to predict
      # the probabilty of using a certain method. Understanding which are most significant could reveal important information about
      # case attributes and their relationships to sucessfully-employed methods
    

    #print case_attrib_bin_matrix_fields #TEST
    #print case_attrib_bin_matrix[:10,] #TEST

    return case_attrib_bin_matrix


def create_all_case_attribs_and_methods_binary_array(case_attrib_bin_matrix, int_matrix):
    '''
    input:  array, case attribute binary matrix (case_attrib_bin_matrix)
            array, interaction matrix, which is binary method data for all cases (int_matrix)
    output:
            array, matrix with all binary attribute data and all binary method data
            
    Function takes as arguments the products of two methods:
       case_attrib_data_to_binary_array()
       create_interaction_matrix()

    Output is ready to use in collaborative filtering
    '''

    return np.concatenate((case_attrib_bin_matrix, int_matrix), axis=1)

    
    
def PLS_regression(matrix_X, matrix_Y):
    '''
    input:  binary arrays, case attributes (matrix_X) and corresponding methods (matrix_Y)
    output: (binary vector, representing method predictions for

    Note: this function is currently unused/unfinished; PLS deprioritized in favor of collaborative filtering
    '''
    components = matrix_X.shape[1] # number of columns (predictor variable/features) in matrix X
    # NOTE: unsure how many components to use...RESEARCH (number of features - i.e. cols in X? number of dep. vars. - i.e. cols in Y?)
    
    pls = PLSRegression(n_components=components)  #original components value was 3 (in example drawn from scikit-learn site)
    pls.fit(matrix_X, matrix_Y)

    #OUTPUT FOR TESTING
    print "Estimated B"
    print np.round(pls.coefs, 2)  # round coefficient values to 2 decimal places
    print "PREDICTED Y"
    print pls.predict(matrix_X)
    print
    print "ACTUAL Y"
    print matrix_Y

    ### <--- LEFT OF HERE. CONTINUE BY TESTING FUNCTION USING MATRICES W/ A KNOWN RELATIONSHIP (Y PREDICTABLE BY X)
    


## Taking and processing user input from website

def user_input_to_binary_vector(user_input):
    '''
    PLACEHOLDER - takes data input by users through RecSys UI on DesEx website (case attributes) and converts to binary vector
    (i.e. a single row, which will be added to the end of the binary array created by case_attrib_data_to_binary_array(), via the function
    add_row_to_binary_matrix()....this matrix will, in turn, be...
        - added to the methods array, via create_all_case_attribs_and_methods_binary_array(), for collaborative filtering
        - or ready to act as matrix_X, for use in PLS Regression, if using
    
    Working assumption: this data will enter via JSON, as array
    '''
    raise NotImplementedError()

def add_row_to_binary_matrix(row, matrix):
    '''
    input:  list, binary vector
    output: operational, adds vector to end of matrix (as last row)

    Tacks on binary vector of user's input (product of function user_input_to_binary_vector())
    '''
    return matrix.append(vector)  # TEST confirm this works
    

if __name__ == "__main__":

##    # TEST DATA
##    X = np.array([[1,0,0,2,1,4,6,0,5,2],
##             [4,0,5,2,1,4,6,0,5,1],
##             [0,0,0,2,1,4,6,1,4,0],
##             [3,0,0,2,1,4,6,0,5,0],
##             [3,0,0,2,1,4,6,0,5,0],
##             [0,3,3,4,4,4,4,5,6,2],
##             [3,0,0,2,1,4,6,0,5,0],
##             [3,6,0,0,0,0,0,0,5,2],])
##    Xb = X.clip(0,1) #clip ound everything above 1 to 1
##    Xb_split = np.hsplit(Xb, [7])    
##    Xb_X, Xb_Y = Xb_split[0], Xb_split[1]
##
##    # Get data from master lists of cases and methods (ids and names for all cases and methods)
##    all_cases, all_methods = get_all_cases_and_methods_data(SQLITE_DB, ALL_CASE_TBL, ALL_CASE_FLD, ALL_METHOD_TBL, ALL_METHOD_FLD)
##    print "ALL CASES ______________________________"
##    print all_cases
##    print
##    print "ALL METHODS _____________________________"
##    print all_methods
##
##    # Get data from cases-methods relationship table (indicating which cases used which methods)
##    cases_to_methods_data = get_cases_to_methods_data(SQLITE_DB, C2M_DB_TBL, C2M_CASE_FLD, C2M_METHOD_FLD)
##    print "CASES TO METHODS _______________________________"
##    print cases_to_methods_data
##
##    # Create interaction matrix (capturing cases-methods relationship table in binary matrix)(for subsequent us in collaborative filtering)
##    int_matrix = create_interaction_matrix(all_cases, all_methods, cases_to_methods_data, ALL_ID_FLD)
##
##    # Create predictions using case-to-case collaborative filter
##    case_to_case_collab_filter_predictions = case_to_case_collab_filter(int_matrix)
##    print "Case-to-case collaborative filtering"
##    print case_to_case_collab_filter_predictions
##
##    # Create predictions using method-to-method collaborative filter
##    method_to_method_collab_filter_predictions = method_to_method_collab_filter(int_matrix)
##    print "Method-to-method collaborative filtering"
##    print method_to_method_collab_filter_predictions
##
##    #PLS TEST
##    #PLS_regression(Xb_X, Xb_Y)
##
##    # Create array with all case attribte data in binary format (as dummy variables)
##    case_attrib_bin_array = case_attrib_data_to_binary_array(SQLITE_DB, ALL_CASE_TBL, CASE_ATTRIB_DICT)
##
##    #TEST - COMPARE CASE ATTRIB BIN ARRAY AND INT MATRIX - make sure they align (for merging in function)
##    print case_attrib_bin_array.shape
##    print int_matrix.shape
##
##    #TEST MERGING FUNCTION OUTPUT
##    big_array = create_all_case_attribs_and_methods_binary_array(case_attrib_bin_array, int_matrix)
##    print "Big array: "
##    print big_array.shape
##    print big_array[:10,:20]
##    big_array_first_two_rows = big_array[:2,]  # get first two rows of big array only
##    big_array_first_two_rows_T = big_array_first_two_rows.T # flip vertically for easier viewing in output
##    print big_array_first_two_rows_T # CHECK THAT ALL ATTRIBS AND METHODS ARE CORRECTLY REP'D IN BIG ARRAY
##
##    #<--- LEFT OFF HERE 2/6 -- CHECK THAT ALL ATTRIBS AND METHODS ARE CORRECTLY REP'D IN BIG ARRAY!!  (once full database is avail)

    

    #TEST - API DATA PULL
    all_cases_api, all_methods_api, cases_to_methods_data_api = get_all_data_via_api(ROOT_API_URL, API_ALL_CASE_TBL_EXT, API_ALL_CASE_FLD, API_ALL_METHOD_TBL_EXT, API_ALL_METHOD_FLD, API_C2M_TBL_EXT, API_C2M_CASE_FLD, API_C2M_METHOD_FLD, ALL_ID_FLD)
    #get_all_data_via_api_light(ROOT_API_URL, API_ALL_CASE_TBL_BY_ID_EXT, API_ALL_METHOD_TBL_BY_ID_EXT, API_C2M_TBL_EXT, API_C2M_CASE_FLD, API_C2M_METHOD_FLD)

    #TEST - TRY FUNCTIONS WITH API DATA
    # Create interaction matrix (capturing cases-methods relationship table in binary matrix)(for subsequent us in collaborative filtering)
    api_int_matrix = create_interaction_matrix(all_cases_api, all_methods_api, cases_to_methods_data_api, ALL_ID_FLD)
    print "all_cases_api"
    print all_cases_api
    print
    print "all_methods_api"
    print all_methods_api
    print
    print "cases_to_methods_data_api"
    print cases_to_methods_data_api
    print api_int_matrix[2,:60]
    

    


##############
## BONEYARD ##
##############
    
##Case id: 16  Method id: 53
##Case id: 1  Method id: 139
##Case id: 17  Method id: 354
##Case id: 9  Method id: 21
##Case id: 10  Method id: 49
##Case id: 11  Method id: 64
##Case id: 12  Method id: 64
##Case id: 12  Method id: 304
##Case id: 13  Method id: 52
##Case id: 13  Method id: 302
##Case id: 14  Method id: 179
##Case id: 15  Method id: 354
##Case id: 18  Method id: 7
##Case id: 75  Method id: 81
##Case id: 80  Method id: 217
##Case id: 81  Method id: 323
##Case id: 100  Method id: 323
##Case id: 100  Method id: 209
##Case id: 104  Method id: 390
##Case id: 105  Method id: 390
##Case id: 105  Method id: 217
##'''

##    print "All cases: "
##    print all_cases
##    print "All methods: "
##    print all_methods
##    print "Cases to methods: "
##    print cases_to_methods_data
##    print len(cases_and_methods_data)                         
##    print cases_and_methods_data[0][1]
    
##    cases = [x[0] for x in cases_to_methods]
##    methods = [x[1] for x in cases_to_methods]
##
##    cases_set = set(cases)
##    methods_set = set(methods)
##    cases_set_len = len(cases_set)
##    methods_set_len = len(methods_set)

        

##    print cases_methods
##    print cases, cases_set
##    print methods, methods_set
##
##    indx = 0
##    for case_unique in cases_set:
##        cases_methods[int(story_id)][int(method_id)]=1    # <--- LEFT OFF HERE
##
##        indx += 1
##
##    # look for better way of acheiving the above (interaction matrix construction)

            
##    for story_id,method_id in case_csv:
##        cases[int(story_id)][int(method_id)]=1


##def case_to_case_collab_filter_2(int_matrix, new_case, k=3, thresh=0.5):
##    '''
##    input:  array, interaction matrix (int_matrix)
##            array, single row (vector) representation of input case
##            int, number of nearest neighbors - i.e. number of 'closest' cases to be used to determine prediced values (k) 
##            float, cutoff probability north of which a method is considered to be probable/recommendable - i.e. marked as '1' in output
##    output: array, binary matrix w/ predicted/recommended values (0 or 1)   
##
##    NOTE:   In refinement stage, consider optimizing k and thresh values (using cross-validation(-like) approach)
##    '''
##    #create distance matrix
##    dist_matrix = dist.squareform(dist.pdist(int_matrix, 'jaccard'))
##
##    #find the closest K cases to all cases in int_matrix
##    closest_matrix = np.argsort(dist_matrix)[:,:k]
##
##    #average the closest K cases
##    result = np.mean(int_matrix[closest_matrix],1)
##    
##    #with threshold average of 0.5, convert all to zeros and ones
##    result = stats.threshold(result,threshmax=thresh,newval=1)
##    result = stats.threshold(result,threshmin=(thresh-0.001),newval=0)
##
##    return result

##    X2 = np.array([[1,0,0,2,1,4,6,0,5,2],
##             [4,0,5,2,1,4,6,0,5,1],
##             [0,0,0,2,1,4,6,1,4,0],
##             [3,0,0,2,1,4,6,0,5,0],
##             [3,0,0,2,1,4,6,0,5,0],
##             [0,3,3,4,4,4,4,5,6,2],
##             [3,0,0,2,1,4,6,0,5,0]])
##    X2_new = np.array([[3,6,0,0,0,0,0,0,5,2]])    
##    X2b = X2.clip(0,1) #clip ound everything above 1 to 1
##    X2_new_b = X2_new.clip(0,1) #clip ound everything above 1 to 1

##
##    print "Xb_split = "
##    print Xb_split
##    print
##    print "Xb_X = "
##    print Xb_X
##    print
##    print "Xb_Y = "
##    print Xb_Y

##    print Xb_X.shape[1]
##    print Xb_Y.shape[1]

 # FROM STACKEXCHANGE (USE!):
##        If we assume you have a DataFrame where some column is 'Category' and contains integers (or otherwise unique identifiers) for categories, then we can do the following.
##
##    Call the DataFrame dfrm, and assume that for each row, dfrm['Category'] is some value in the set of integers from 1 to N. Then,
##
##    for elem in dfrm['Category'].unique():
##        dfrm[str(elem)] = dfrm['Category'] == elem
##
##    Now there will be a new indicator column for each category that is True/False depending on whether the data in that row are in that category.
##
##    If you want to control the category names, you could make a dictionary, such as
##
##    cat_names = {1:'Some_Treatment', 2:'Full_Treatment', 3:'Control'}
##    for elem in dfrm['Category'].unique():
##        dfrm[cat_names[elem]] = dfrm['Category'] == elem
##
##    to result in having columns with specified names, rather than just

 
##    def whatisthis(s):
##        if isinstance(s, str):
##            print "ordinary string"
##        elif isinstance(s, unicode):
##            print "unicode string"
##        else:
##            print "not a string"string conversion of the category values. In fact, for some types, str() may not produce anything useful for you.
##

##                   
####    for fld in bool_fld_list:
####        df.applymap(lambda x: x=='t')
##
##        
##        df[bool_fld_list + "_bool"]
##        for elem in df[fld]:
##            df[
##
##    for fld in bool_fld_list:
##        for elem in df[fld]:
##            elem.apply(to_Bool)
#
##    # convert all values to 0s and 1s
##    for fld in dummy_fld_list:          # dummy fields
##        for elem in df[fld]:
##            if elem == True:
##                df[fld][elem] = 1
##            elif elem == False:
##                df[fld][elem] = 0
##                    
##    for fld in dummy_fld_list:          # boolean fields
##        for elem in df[fld]:
##            if elem == "t":
##                df[fld][elem] = 1
##            elif elem == "f":
##                df[fld][elem] = 0
##
##    print df
##
##    # CLEAN DATASET:
##    # keep only rows (cases) with minimum 4 non-NaN values
##    # Consider allowing ONLY cases with ALL attributes filled out in RecSys
##    # (to do this, replace "thresh" argument with "how" and set to "any" (as str))    
##    df = df.dropna(axis=0, thresh=4)


# OJO! -> matrix row numbers correspond to (case_id - 1) and column numbers to (method_id - 1)
##    print interaction_matrix
##    print interaction_matrix[15][52]
##    print interaction_matrix[0][138]




####    url = "https://www.googleapis.com/qpxExpress/v1/trips/search?key=mykeyhere"
####    my_json_data = json.load(open("request.json"))
####    req = request.post(url,data=my_json_data)
####    print req.text
####    print 
####    print req.json # maybe? 
##
##    # Get and process data at each request url
##    # case table
##    print "Executing requests.get(case_tbl_url)..."
##    data = requests.get(case_tbl_url)       # Get data object via api call
##    print "data"
##    print data
##    print "type(data)"
##    print type(data)
##    
##    #data_json = json.loads(data.text)
##    data_json = data.json()                     # Get json representation of data object
##    print "data_json[0]"
##    print data_json[0]
##    #data_json_py = json.loads(str(data_json))   # Convert json to python object
##                                                # Convert python object to array
##
##    # TEST
##    print "dat_json type:"
##    print type(data_json)
##    #print "data_json_py"
##    #print data_json_py
##    #print "type(data_json_py)"
##    #print type(data_json_py)
##    #print "data_json contents:"
##    #print data_json
##    print "json.dumps"
##    print json.dumps(data_json)
##
####    import pycurl
####    import pprint
####
####    c = pycurl.Curl()
####    c.setopt(c.URL, case_tbl_url)
####
####    result = c.perform()
####
####    print result
##                
##
####    my_data = json.loads(open("xyz.json").read())
####
####    print my_data



### TEST - DELETE
##def api_test():
##    r = requests.get('https://api.github.com/events')
##    r_json = r.json()
##
##    print "type(r_json)"
##    print type(r_json)
##    print "r_json"
##    print r_json[:2]
##    
####    r_json_py = json.loads(str(r_json))
####    print "type(r_json_py)"
####    print type(r_json_py)
####    print "r_json_py"
####    print r_json_py[:2]


##    case_tbl_data_clean_sorted = sorted(case_tbl_data_clean, key=lambda case: case[0])
##    print "case_tbl_data_clean_sorted"
##    print case_tbl_data_clean_sorted


# Using 'requests' module
# see http://docs.python-requests.org/en/latest/user/quickstart/

# From Design Exchange Wiki

# either send a get request with Accept: application/json header or send a get request to /path/file.json
# all design methods: /design_methods
# design methods by ID: /design_methods/id
# all case studies: GET /case_studies
# case studies by ID: /case_studies/id
# Method case studies: /method_case_studies


##def get_all_data_via_api_light(url, case_tbl_by_id_ext, method_tbl_by_id_ext, c2m_tbl_ext, c2m_case_fld, c2m_method_fld):
##
##    '''
##
##    CURRENTLY NOT WORKING
##    
##    input:  str, root DesEx url (url)
##            ... 
##            (str, name of target database table (method-case study relation) (tbl))
##            (str, name of field with cases in target db (case_fld))
##            (str, name of field with methods in target db (method_fld))
##            ...
##    output: three arrays:
##                (1) all methods, w/ two cols: id, name
##                (2) all cases, w/ two cols: id, name
##                (3) case-to-methods, w/ two cols: case_id, method_id
##
##    dependencies: requests, json
##
##    NOTE: If it is necessary to pass in parameters to API call - that is, include params in url, use as template:
##        payload = {'key1': 'value1', 'key2': 'value2'}
##        r = requests.get("http://httpbin.org/get", params=payload)
##
##    '''
##    # Construct request urls
##    case_tbl_by_id_url = url + case_tbl_by_id_ext #+ ".json"
##    method_tbl_by_id_url = url + method_tbl_by_id_ext #+ ".json"
##    c2m_tbl_url = url + c2m_tbl_ext #+ ".json"
##    
##    # TEST
##    print "case_tbl_by_id_url"
##    print case_tbl_by_id_url
##    print "method_tbl_by_id_url"
##    print method_tbl_by_id_url
##    print "c2m_tbl_url"
##    print c2m_tbl_url
##    print "getting data via API..."
##
##    # Get data via urls
##    case_tbl_data = pd.io.json.read_json(case_tbl_by_id_url)
##    method_tbl_data = pd.io.json.read_json(method_tbl_by_id_url)
##    c2m_tbl_data = pd.io.json.read_json(c2m_tbl_url)
##
##    # Convert rec'd json data to numpy arrays, removing all but columns to keep   
##    case_tbl_data_clean = case_tbl_dat.as_matrix(columns = None)
##    method_tbl_data_clean = method_tbl_data.as_matrix(columns = None)
##    c2m_tbl_data_clean = c2m_tbl_data.as_matrix(columns = None)
##
##    print "case_tbl_data_clean[:5,]"
##    print case_tbl_data_clean[:5,]
##    print "method_tbl_data_clean[:5,]"
##    print method_tbl_data_clean[:5,]
##    print "c2m_tbl_data_clean[:5,]"
##    print c2m_tbl_data_clean[:5,]
##    
##    #return case_tlb_data_clean, method_tbl_data_clean, c2m_tbl_data_clean
