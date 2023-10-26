#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>
#include <algorithm>
#include <math.h>
#include <random>
#include <chrono>
#include <Eigen/Dense>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Vector3d;

// vector<float> sepal_length;
// vector<float> sepal_width;
// vector<float> petal_length;
// vector<float> petal_width;
// // vector<vector<float>> observation;
// vector<vector<float>> observation_zero_mean;
// vector<vector<float>> gaussian_mean;
// vector<float> prior_probability;
// vector<vector<float>> gaussian_deviation;
// vector<vector<float>> covariance_gaussian1;
// vector<vector<float>> covariance_gaussian2;
// vector<vector<float>> covariance_gaussian3;
int gaussians = 3;

MatrixXd observation(150, 5);
MatrixXd observation_1(50, 4);
MatrixXd observation_2(50, 4);
MatrixXd observation_3(50, 4);
VectorXd mean_gaussian1;
VectorXd mean_gaussian2;
VectorXd mean_gaussian3;
VectorXd std_dev_gaussian1;
VectorXd std_dev_gaussian2;
VectorXd std_dev_gaussian3;
Vector3d prior_probability;
MatrixXd gaussian1_covariance(4,4);
MatrixXd gaussian2_covariance(4,4);
MatrixXd gaussian3_covariance(4,4);

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
            {
                observation(obs_no, params) = stod(data);
            }
            else if (params == 4)
            {
                if (data.compare("Iris-setosa") == 0)
                    observation(obs_no, params) = 0;
                else if (data.compare("Iris-versicolor") == 0)
                    observation(obs_no, params) = 1;
                else
                    observation(obs_no, params) = 2;
            }

            params++;
        }
        // observation.push_back(data_point);
        obs_no++;
    }
}
void assign_points_to_gaussians()
{
    for(int i=0; i<50; i++)
    {
        for(int j=0; j<4; j++)
        {
            observation_1(i,j) = observation(i,j);
        }
    }

    for(int i=50; i<100; i++)
    {
        for(int j=0; j<4; j++)
        {
            observation_2(i - 50,j) = observation(i,j);
        }
    }

    for(int i=100; i<150; i++)
    {
        for(int j=0; j<4; j++)
        {
            observation_3(i - 100,j) = observation(i,j);
        }
    }
}
void get_gaussian_initial_mean_stddev()
{

    // getting mean of 4 attributes for 0-49 observations
    mean_gaussian1 = observation_1.colwise().mean();
    
    // getting mean of 4 attributes for 50-99 observations
    mean_gaussian2 = observation_2.colwise().mean();

    // getting mean of 4 attributes for 100-149 observations
    mean_gaussian3 = observation_3.colwise().mean();

    

    // centering data points at origin
    observation_1 = observation_1.rowwise() - mean_gaussian1.transpose();
    observation_2 = observation_2.rowwise() - mean_gaussian2.transpose();
    observation_3 = observation_3.rowwise() - mean_gaussian3.transpose();

    // squaring all the data columnwise dividing by (n-1) and finding sqrt to finally get std_deviation

    std_dev_gaussian1 = observation_1.colwise().squaredNorm();
    std_dev_gaussian1 = std_dev_gaussian1/(50 - 1);
    std_dev_gaussian1 = std_dev_gaussian1.transpose().cwiseSqrt().transpose();

    std_dev_gaussian2 = observation_2.colwise().squaredNorm();
    std_dev_gaussian2 = std_dev_gaussian2/(50 - 1);
    std_dev_gaussian2 = std_dev_gaussian2.transpose().cwiseSqrt().transpose();

    std_dev_gaussian3 = observation_3.colwise().squaredNorm();
    std_dev_gaussian3 = std_dev_gaussian3/(50 - 1);  
    std_dev_gaussian3 = std_dev_gaussian3.transpose().cwiseSqrt().transpose();
    
    
}

void initialize_gaussian_priors()
{
    // initializing priors for each gaussian

    Vector3d prior;
    float rand_sum=0;
    int lb = 1, ub = 10;
    srand(time(0));
    // numbers on every program run within range lb to ub
    for (int i = 0; i < 3; i++)
    {
        float rand_num = ((rand() % (ub - lb + 1)) + lb );
        prior(i,0) = rand_num;
        rand_sum += rand_num;
    }
    // prior.resize(3);
    // for(int i=0;i<3;i++)
    // {
    //     prior[i] = prior[i]/rand_sum;
    //     prior_probability.push_back(prior[i]);
    // }
    prior = prior/rand_sum;
    prior_probability = prior;
    

}

void get_initial_covariance(MatrixXd m1, int gaussian)
{
    
    for(int i=0;i<4;i++)
    {   
        
        
        for(int j=0;j<4;j++)
        {
            switch (gaussian)
            {
            case 1:
                gaussian1_covariance(i, j) = m1.col(i).dot(m1.col(j));
                break;
            case 2:
                gaussian2_covariance(i, j) = m1.col(i).dot(m1.col(j));
                break;
            case 3:
                gaussian3_covariance(i, j) = m1.col(i).dot(m1.col(j));
                break;
            default:
                break;
            }
            
            
        }
        
    }

    if(gaussian==1)
        gaussian1_covariance /= (50 - 1);
    else if(gaussian==2)
        gaussian2_covariance /= (50 - 1);
    else
        gaussian3_covariance /= (50 - 1);
}

void shuffle_data()
{
    srand(time(0));
    for (int i = 0; i < 150 - 1; i++)
    {
        int j = i + rand() % (150 - i);
        observation.row(i).swap(observation.row(j));
    }
}
int main()
{
    // read data from csv file
    read_csv_data("iris.txt");
    // shuffling the observations
    
    shuffle_data();

    // initializing the parameters of the gaussians randomly
    
    assign_points_to_gaussians();

    get_gaussian_initial_mean_stddev();

    initialize_gaussian_priors();

    cout<<observation<<endl;
    get_initial_covariance(observation_1, 1);
    get_initial_covariance(observation_2, 2);
    get_initial_covariance(observation_3, 3);

    // cout<<"Covariance matrices"<<endl;
    // cout<<gaussian1_covariance<<endl;
    // cout<<gaussian2_covariance<<endl;
    // cout<<gaussian3_covariance<<endl;
    return 0;
}