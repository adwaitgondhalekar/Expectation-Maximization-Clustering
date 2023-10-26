#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>
#include <algorithm>
#include <math.h>
#include <random>
#include <chrono>

using namespace std;

vector<float> sepal_length;
vector<float> sepal_width;
vector<float> petal_length;
vector<float> petal_width;
vector<vector<float>> observation;
vector<vector<float>> observation_zero_mean;
vector<vector<float>> gaussian_mean;
vector<float> prior_probability;
vector<vector<float>> gaussian_deviation;
vector<vector<float>> covariance_gaussian1;
vector<vector<float>> covariance_gaussian2;
vector<vector<float>> covariance_gaussian3;
int gaussians = 3;

void read_csv_data(string filename)
{

    fstream file;
    string line;
    file.open(filename, ios::in | ios::out);
    int obs_no = 0;
    while (file)
    {

        getline(file, line);
        istringstream is(line);
        int params = 0;
        string data = "";
        vector<float> data_point;
        while (getline(is, data, ','))
        {
            if (params < 4)
                data_point.push_back(stof(data));
            params++;
        }
        observation.push_back(data_point);
        obs_no++;
    }
}
vector<float> get_mean_obs(int start, int end)
{
    float sum_attr0 = 0;
    float sum_attr1 = 0;
    float sum_attr2 = 0;
    float sum_attr3 = 0;
    float total_obs = end - start;
    vector<float> attribute_wise_means;
    for (int i = start; i < end; i++)
    {
        sum_attr0 += observation[i][0];
        sum_attr1 += observation[i][1];
        sum_attr2 += observation[i][2];
        sum_attr3 += observation[i][3];
    }

    attribute_wise_means.push_back((sum_attr0 / total_obs));
    attribute_wise_means.push_back((sum_attr1 / total_obs));
    attribute_wise_means.push_back((sum_attr2 / total_obs));
    attribute_wise_means.push_back((sum_attr3 / total_obs));

    return attribute_wise_means;
}

vector<float> get_std_deviation(int start, int end, int gaussian)
{
    float zero_mean_attr0_sq_sum = 0;
    float zero_mean_attr1_sq_sum = 0;
    float zero_mean_attr2_sq_sum = 0;
    float zero_mean_attr3_sq_sum = 0;
    float total_obs = end - start;
    vector <float> attr_wise_std_dev;

    // subtracting the atribute_mean from each attribute of observations

    for(int i=start;i<end;i++)
    {   
        vector <float> zero_mean_value;
        for(int j=0;j<4;j++)
        {
            zero_mean_value.push_back(observation[i][j] - gaussian_mean[j][gaussian - 1]);
        }
        observation_zero_mean.push_back(zero_mean_value);
        zero_mean_value.clear();
    }

    // squaring the new zero mean observations
    for(int i=start;i<end;i++)
    {
        zero_mean_attr0_sq_sum += pow(observation_zero_mean[i][0],2);
        zero_mean_attr1_sq_sum += pow(observation_zero_mean[i][1],2);
        zero_mean_attr2_sq_sum += pow(observation_zero_mean[i][2],2);
        zero_mean_attr3_sq_sum += pow(observation_zero_mean[i][3],2);

    }

    attr_wise_std_dev.push_back(sqrtf((zero_mean_attr0_sq_sum/(total_obs - 1))));
    attr_wise_std_dev.push_back(sqrtf((zero_mean_attr1_sq_sum/(total_obs - 1))));
    attr_wise_std_dev.push_back(sqrtf((zero_mean_attr2_sq_sum/(total_obs - 1))));
    attr_wise_std_dev.push_back(sqrtf((zero_mean_attr3_sq_sum/(total_obs - 1))));

    return attr_wise_std_dev;
}

void get_covariance_gaussian1(int start, int end)
{
    float covar_sl_sl = 0;
    float covar_sl_sw = 0;
    float covar_sl_pl = 0;
    float covar_sl_pw = 0;
    float covar_sw_sw = 0;
    float covar_sw_pl = 0;
    float covar_sw_pw = 0;
    float covar_pl_pl = 0;
    float covar_pl_pw = 0;
    float covar_pw_pw = 0;
    int total_obs = end-start;
    vector <float> row_of_covar_matrix;
    for(int i=start;i<end;i++)
    {
        covar_sl_sl += observation_zero_mean[i][0]*observation_zero_mean[i][0];
        covar_sl_sw += observation_zero_mean[i][0]*observation_zero_mean[i][1];
        covar_sl_pl += observation_zero_mean[i][0]*observation_zero_mean[i][2];
        covar_sl_pw += observation_zero_mean[i][0]*observation_zero_mean[i][3];
        covar_sw_sw += observation_zero_mean[i][1]*observation_zero_mean[i][1];
        covar_sw_pl += observation_zero_mean[i][1]*observation_zero_mean[i][2];
        covar_sw_pw += observation_zero_mean[i][1]*observation_zero_mean[i][3];
        covar_pl_pl += observation_zero_mean[i][2]*observation_zero_mean[i][2];
        covar_pl_pw += observation_zero_mean[i][2]*observation_zero_mean[i][3];
        covar_pw_pw += observation_zero_mean[i][3]*observation_zero_mean[i][3];
    }

    covar_sl_sl /= (total_obs-1);
    covar_sl_sw /= (total_obs-1);
    covar_sl_pl /= (total_obs-1);
    covar_sl_pw /= (total_obs-1);
    covar_sw_sw /= (total_obs-1);
    covar_sw_pl /= (total_obs-1);
    covar_sw_pw /= (total_obs-1);
    covar_pl_pl /= (total_obs-1);
    covar_pl_pw /= (total_obs-1);
    covar_pw_pw /= (total_obs-1);

    row_of_covar_matrix.push_back(covar_sl_sl);
    row_of_covar_matrix.push_back(covar_sl_sw);
    row_of_covar_matrix.push_back(covar_sl_pl);
    row_of_covar_matrix.push_back(covar_sl_pw);


    covariance_gaussian1.push_back(row_of_covar_matrix);

    row_of_covar_matrix.clear();

    row_of_covar_matrix.push_back(covar_sl_sw);
    row_of_covar_matrix.push_back(covar_sw_sw);
    row_of_covar_matrix.push_back(covar_sw_pl);
    row_of_covar_matrix.push_back(covar_sw_pw);

    covariance_gaussian1.push_back(row_of_covar_matrix);
    row_of_covar_matrix.clear();


    row_of_covar_matrix.push_back(covar_sl_pl);
    row_of_covar_matrix.push_back(covar_sw_pl);
    row_of_covar_matrix.push_back(covar_pl_pl);
    row_of_covar_matrix.push_back(covar_pl_pw);

    covariance_gaussian1.push_back(row_of_covar_matrix);
    row_of_covar_matrix.clear();


    row_of_covar_matrix.push_back(covar_sl_pw);
    row_of_covar_matrix.push_back(covar_sw_pw);
    row_of_covar_matrix.push_back(covar_pl_pw);
    row_of_covar_matrix.push_back(covar_pw_pw);


    covariance_gaussian1.push_back(row_of_covar_matrix);
    
    
}

void get_covariance_gaussian2(int start, int end)
{
    float covar_sl_sl = 0;
    float covar_sl_sw = 0;
    float covar_sl_pl = 0;
    float covar_sl_pw = 0;
    float covar_sw_sw = 0;
    float covar_sw_pl = 0;
    float covar_sw_pw = 0;
    float covar_pl_pl = 0;
    float covar_pl_pw = 0;
    float covar_pw_pw = 0;
    int total_obs = end-start;
    vector <float> row_of_covar_matrix;
    for(int i=start;i<end;i++)
    {
        covar_sl_sl += observation_zero_mean[i][0]*observation_zero_mean[i][0];
        covar_sl_sw += observation_zero_mean[i][0]*observation_zero_mean[i][1];
        covar_sl_pl += observation_zero_mean[i][0]*observation_zero_mean[i][2];
        covar_sl_pw += observation_zero_mean[i][0]*observation_zero_mean[i][3];
        covar_sw_sw += observation_zero_mean[i][1]*observation_zero_mean[i][1];
        covar_sw_pl += observation_zero_mean[i][1]*observation_zero_mean[i][2];
        covar_sw_pw += observation_zero_mean[i][1]*observation_zero_mean[i][3];
        covar_pl_pl += observation_zero_mean[i][2]*observation_zero_mean[i][2];
        covar_pl_pw += observation_zero_mean[i][2]*observation_zero_mean[i][3];
        covar_pw_pw += observation_zero_mean[i][3]*observation_zero_mean[i][3];
    }

    covar_sl_sl /= (total_obs-1);
    covar_sl_sw /= (total_obs-1);
    covar_sl_pl /= (total_obs-1);
    covar_sl_pw /= (total_obs-1);
    covar_sw_sw /= (total_obs-1);
    covar_sw_pl /= (total_obs-1);
    covar_sw_pw /= (total_obs-1);
    covar_pl_pl /= (total_obs-1);
    covar_pl_pw /= (total_obs-1);
    covar_pw_pw /= (total_obs-1);

    row_of_covar_matrix.push_back(covar_sl_sl);
    row_of_covar_matrix.push_back(covar_sl_sw);
    row_of_covar_matrix.push_back(covar_sl_pl);
    row_of_covar_matrix.push_back(covar_sl_pw);


    covariance_gaussian2.push_back(row_of_covar_matrix);

    row_of_covar_matrix.clear();

    row_of_covar_matrix.push_back(covar_sl_sw);
    row_of_covar_matrix.push_back(covar_sw_sw);
    row_of_covar_matrix.push_back(covar_sw_pl);
    row_of_covar_matrix.push_back(covar_sw_pw);

    covariance_gaussian2.push_back(row_of_covar_matrix);
    row_of_covar_matrix.clear();


    row_of_covar_matrix.push_back(covar_sl_pl);
    row_of_covar_matrix.push_back(covar_sw_pl);
    row_of_covar_matrix.push_back(covar_pl_pl);
    row_of_covar_matrix.push_back(covar_pl_pw);

    covariance_gaussian2.push_back(row_of_covar_matrix);
    row_of_covar_matrix.clear();


    row_of_covar_matrix.push_back(covar_sl_pw);
    row_of_covar_matrix.push_back(covar_sw_pw);
    row_of_covar_matrix.push_back(covar_pl_pw);
    row_of_covar_matrix.push_back(covar_pw_pw);


    covariance_gaussian2.push_back(row_of_covar_matrix);
    

}

void get_covariance_gaussian3(int start, int end)
{
    float covar_sl_sl = 0;
    float covar_sl_sw = 0;
    float covar_sl_pl = 0;
    float covar_sl_pw = 0;
    float covar_sw_sw = 0;
    float covar_sw_pl = 0;
    float covar_sw_pw = 0;
    float covar_pl_pl = 0;
    float covar_pl_pw = 0;
    float covar_pw_pw = 0;
    int total_obs = end-start;
    vector <float> row_of_covar_matrix;
    for(int i=start;i<end;i++)
    {
        covar_sl_sl += observation_zero_mean[i][0]*observation_zero_mean[i][0];
        covar_sl_sw += observation_zero_mean[i][0]*observation_zero_mean[i][1];
        covar_sl_pl += observation_zero_mean[i][0]*observation_zero_mean[i][2];
        covar_sl_pw += observation_zero_mean[i][0]*observation_zero_mean[i][3];
        covar_sw_sw += observation_zero_mean[i][1]*observation_zero_mean[i][1];
        covar_sw_pl += observation_zero_mean[i][1]*observation_zero_mean[i][2];
        covar_sw_pw += observation_zero_mean[i][1]*observation_zero_mean[i][3];
        covar_pl_pl += observation_zero_mean[i][2]*observation_zero_mean[i][2];
        covar_pl_pw += observation_zero_mean[i][2]*observation_zero_mean[i][3];
        covar_pw_pw += observation_zero_mean[i][3]*observation_zero_mean[i][3];
    }

    covar_sl_sl /= (total_obs-1);
    covar_sl_sw /= (total_obs-1);
    covar_sl_pl /= (total_obs-1);
    covar_sl_pw /= (total_obs-1);
    covar_sw_sw /= (total_obs-1);
    covar_sw_pl /= (total_obs-1);
    covar_sw_pw /= (total_obs-1);
    covar_pl_pl /= (total_obs-1);
    covar_pl_pw /= (total_obs-1);
    covar_pw_pw /= (total_obs-1);

    row_of_covar_matrix.push_back(covar_sl_sl);
    row_of_covar_matrix.push_back(covar_sl_sw);
    row_of_covar_matrix.push_back(covar_sl_pl);
    row_of_covar_matrix.push_back(covar_sl_pw);


    covariance_gaussian3.push_back(row_of_covar_matrix);

    row_of_covar_matrix.clear();

    row_of_covar_matrix.push_back(covar_sl_sw);
    row_of_covar_matrix.push_back(covar_sw_sw);
    row_of_covar_matrix.push_back(covar_sw_pl);
    row_of_covar_matrix.push_back(covar_sw_pw);

    covariance_gaussian3.push_back(row_of_covar_matrix);
    row_of_covar_matrix.clear();


    row_of_covar_matrix.push_back(covar_sl_pl);
    row_of_covar_matrix.push_back(covar_sw_pl);
    row_of_covar_matrix.push_back(covar_pl_pl);
    row_of_covar_matrix.push_back(covar_pl_pw);

    covariance_gaussian3.push_back(row_of_covar_matrix);
    row_of_covar_matrix.clear();


    row_of_covar_matrix.push_back(covar_sl_pw);
    row_of_covar_matrix.push_back(covar_sw_pw);
    row_of_covar_matrix.push_back(covar_pl_pw);
    row_of_covar_matrix.push_back(covar_pw_pw);


    covariance_gaussian3.push_back(row_of_covar_matrix);
    

}
void initialize_gaussians()
{
    vector<float> mean_attr0;
    vector<float> mean_attr1;
    vector<float> mean_attr2;
    vector<float> mean_attr3;
    vector<float> std_dev_attr0;
    vector<float> std_dev_attr1;
    vector<float> std_dev_attr2;
    vector<float> std_dev_attr3;

    // getting attribute wise mean for batch of 0-49 observations (for gaussian 1)
    vector<float> attr_wise_means = get_mean_obs(0, 50);

    mean_attr0.push_back(attr_wise_means[0]);
    mean_attr1.push_back(attr_wise_means[1]);
    mean_attr2.push_back(attr_wise_means[2]);
    mean_attr3.push_back(attr_wise_means[3]);

    attr_wise_means.clear();


    // getting attribute wise mean for batch of 50-99 observations (for gaussian 2)
    attr_wise_means = get_mean_obs(50, 100);
    mean_attr0.push_back(attr_wise_means[0]);
    mean_attr1.push_back(attr_wise_means[1]);
    mean_attr2.push_back(attr_wise_means[2]);
    mean_attr3.push_back(attr_wise_means[3]);

    attr_wise_means.clear();


    // getting attribute wise mean for batch of 100-149 observations (for gaussian 3)
    attr_wise_means = get_mean_obs(100, 150);
    mean_attr0.push_back(attr_wise_means[0]);
    mean_attr1.push_back(attr_wise_means[1]);
    mean_attr2.push_back(attr_wise_means[2]);
    mean_attr3.push_back(attr_wise_means[3]);

    gaussian_mean.push_back(mean_attr0);
    gaussian_mean.push_back(mean_attr1);
    gaussian_mean.push_back(mean_attr2);
    gaussian_mean.push_back(mean_attr3);

    // initializing std deviation for each gaussian

    // getting attribute wise std dev for batch of 0-49 observations (for gaussian 1)
    vector<float> attr_wise_std_dev = get_std_deviation(0, 50, 1);
    std_dev_attr0.push_back(attr_wise_std_dev[0]);
    std_dev_attr1.push_back(attr_wise_std_dev[1]);
    std_dev_attr2.push_back(attr_wise_std_dev[2]);
    std_dev_attr3.push_back(attr_wise_std_dev[3]);

    attr_wise_std_dev.clear();

    // getting attribute wise std dev for batch of 50-99 observations (for gaussian 2)
    attr_wise_std_dev = get_std_deviation(50, 100, 2);
    std_dev_attr0.push_back(attr_wise_std_dev[0]);
    std_dev_attr1.push_back(attr_wise_std_dev[1]);
    std_dev_attr2.push_back(attr_wise_std_dev[2]);
    std_dev_attr3.push_back(attr_wise_std_dev[3]);

    attr_wise_std_dev.clear();


    // getting attribute wise std dev for batch of 100-149 observations (for gaussian 3)
    attr_wise_std_dev = get_std_deviation(100, 150, 3);
    std_dev_attr0.push_back(attr_wise_std_dev[0]);
    std_dev_attr1.push_back(attr_wise_std_dev[1]);
    std_dev_attr2.push_back(attr_wise_std_dev[2]);
    std_dev_attr3.push_back(attr_wise_std_dev[3]);

    gaussian_deviation.push_back(std_dev_attr0);
    gaussian_deviation.push_back(std_dev_attr1);
    gaussian_deviation.push_back(std_dev_attr2);
    gaussian_deviation.push_back(std_dev_attr3);


    // initializing the co-variance matrix for each gaussian
    get_covariance_gaussian1(0,50);
    get_covariance_gaussian2(50,100);
    get_covariance_gaussian3(100,150);
    

    // initializing priors for each gaussian

    vector<float> prior;
    float rand_sum=0;
    int lb = 1, ub = 10; 
    
    // numbers on every program run within range lb to ub 
    for (int i = 0; i < 3; i++)
    {
        float rand_num = ((rand() % (ub - lb + 1)) + lb );
        prior.push_back(rand_num);
        rand_sum += rand_num;
    }
    prior.resize(3);
    for(int i=0;i<3;i++)
    {
        prior[i] = prior[i]/rand_sum;
        prior_probability.push_back(prior[i]);
    }
        
}

void e_step()
{

}

int main()
{
    // read data from csv file
    read_csv_data("iris.txt");
    observation.resize(150);

    // shuffling the observations
    // obtain a time-based seed:

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

    shuffle (observation.begin(), observation.end(), std::default_random_engine(seed));


    

    // initializing the parameters of the gaussians randomly
    initialize_gaussians();

    // cout<<"gaussian means"<<endl;
    // for(int i=0;i<4;i++)
    // {
    //     for(int j=0;j<gaussian_mean[i].size();j++)
    //     {
    //         cout<<gaussian_mean[i][j]<<"\t";
    //     }
    //     cout<<endl;
    // }

    // cout<<"actual observations"<<endl;
    // for(int i=0;i<150;i++)
    // {
    //     for(int j=0;j<observation[i].size();j++)
    //     {
    //         cout<<observation[i][j]<<"\t";
    //     }
    //     cout<<endl;
    // } 

    // cout<<"after zero mean"<<endl;
    // for(int i=0;i<150;i++)
    // {
    //     for(int j=0;j<observation_zero_mean[i].size();j++)
    //     {
    //         cout<<observation_zero_mean[i][j]<<"\t";
    //     }
    //     cout<<endl;
    // }

    // cout<<"gaussians std dev"<<endl;
    // for(int i=0;i<4;i++)
    // {
    //     for(int j=0;j<gaussian_deviation[i].size();j++)
    //     {
    //         cout<<gaussian_deviation[i][j]<<"\t";
    //     }
    //     cout<<endl;
    // }

    // cout<<"prior probabilities"<<endl;

    // for(int i=0;i<3;i++)
    // {
    //     cout<<prior_probability[i]<<"\t";
    // }
    cout<<"covariance for gaussian1"<<endl;
    for(int i=0;i<4;i++)
    {
        for(int j=0;j<4;j++)
        {
            cout<<covariance_gaussian1[i][j]<<"\t";
        }
        cout<<endl;
    }
    cout<<"covariance for gaussian2"<<endl;
    for(int i=0;i<4;i++)
    {
        for(int j=0;j<4;j++)
        {
            cout<<covariance_gaussian2[i][j]<<"\t";
        }
        cout<<endl;
    }
    cout<<"covariance for gaussian3"<<endl;
    for(int i=0;i<4;i++)
    {
        for(int j=0;j<4;j++)
        {
            cout<<covariance_gaussian3[i][j]<<"\t";
        }
        cout<<endl;
    }
    // e_step();
    return 0;
}