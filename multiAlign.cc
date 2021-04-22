#include <iostream>
#include <fstream>
#include <sstream>
#include "ceres/ceres.h"
#include "glog/logging.h"

using ceres::AutoDiffCostFunction;
using ceres::DynamicAutoDiffCostFunction;
using ceres::CostFunction;
using ceres::CauchyLoss;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;
using namespace std;

typedef vector<int> VecInt;
typedef vector<VecInt> MatInt;
static constexpr int kStride = 100;

struct Cluster{
    int label;
    VecInt indices;
};
struct Corres{
    int id0;
    int id1;
    double dist;
};
struct DayPair{
    int day0;
    int day1;
    vector<Corres> corres_data;
};

class Ronin{
    public:
        ~Ronin(){
            delete[] mags;
            delete[] thetas;
            //delete[] biases;
        }

        //double* get_biases() {return biases;}
        vector<Cluster>& get_cluster_infos() {return cluster_infos;}
        //VecInt& get_cluster_indices(int i) {return cluster_indices[i];}
        double* get_mags() {return mags;}
        double* get_thetas() {return thetas;}
        int get_n_velocity() {return n_velocity;}
        double* get_gtransforms() {return gtransfroms;}

        bool write_mag_theta(string &filename){
        	ofstream fout(filename);
        	fout<<n_velocity<<endl;
        	double updated_x=gtransfroms[0], updated_y=gtransfroms[1];
        	for (int i=0; i<n_velocity; i++){
        		updated_x += mags[i]*cos(thetas[i]+gtransfroms[2]);
        		updated_y += mags[i]*sin(thetas[i]+gtransfroms[2]);
        		fout<<mags[i]<<"\t"<<thetas[i]<<"\t"<<gtransfroms[2]<<"\t"<<updated_x<<"\t"<<updated_y<<endl;
        	}
            fout<<gtransfroms[0]<<"\t"<<gtransfroms[1]<<endl;
        	fout.close();
        	return true;
        }

        bool read_mag_theta(string &filename){//read single align result
            ifstream fin;
            // read mag and theta
            fin.open(filename);
            if (!fin){
                cerr<<"cannot read mag theta"<<endl;
                exit(1);
            }
            string tmp;
            fin>>n_velocity;
            mags = new double[n_velocity];
            thetas = new double[n_velocity];
            double bias;
            for (int i=0; i<n_velocity; i++){
                fin>>mags[i]>>thetas[i]>>bias>>tmp>>tmp;
                thetas[i] += bias;
            }
            fin.close();
            // for (int i=0; i<30; i++){
            //     cout<<mags[i]<<"\t"<<thetas[i]<<endl;
            // }
            return true;
        }
        bool read_cluster_infos(string &filename){
            ifstream fin;
            // read mag and theta
            fin.open(filename);
            if (!fin){
                cerr<<"cannot read cluster indices"<<endl;
                exit(1);
            }
            int tmp;
            fin>>tmp;
            cluster_infos.resize(tmp);
            for (size_t i=0;i<cluster_infos.size();i++){
                fin>>cluster_infos[i].label>>tmp;
                cluster_infos[i].indices.resize(tmp);
                for (int j=0;j<tmp;j++){
                    fin>>cluster_infos[i].indices[j];
                }
            }
            fin.close();
            // print
            // for (size_t i=0;i<cluster_indices.size();i++){
            //     for (size_t j=0;j<cluster_indices[i].size();j++){
            //         cout<<cluster_indices[i][j]<<"\t";
            //     }
            //     cout<<endl;
            // }
            return true;
        }

    private:
        int n_velocity;
        double* mags;
        double* thetas;
        vector<Cluster> cluster_infos;
        //double* biases;
        double gtransfroms[3] = {0.}; //global transform (startx, starty, global biase)

};

struct CorresConstraint {
    typedef DynamicAutoDiffCostFunction<CorresConstraint, kStride>
        CorresCostFunction;

    CorresConstraint(size_t total_num_corres,
                    vector<Ronin>* all_ronins,
                    vector<DayPair>* day_pairs)
        : total_num_corres(total_num_corres),
          all_ronins(all_ronins),
          day_pairs(day_pairs) {}

    template <typename T>
    bool operator()(T const* const* params, T* residuals) const {
        vector<vector<T> > updated_x((*all_ronins).size());
        vector<vector<T> > updated_y((*all_ronins).size());
        size_t pid = 0;
        for (size_t d=0;d<(*all_ronins).size();d++){
            double* mags = (*all_ronins)[d].get_mags();
            double* thetas = (*all_ronins)[d].get_thetas();
            int n_velocity = (*all_ronins)[d].get_n_velocity();
            T xstart= T(params[pid][0]), ystart = T(params[pid+1][0]);
            T gbias = T(params[pid+2][0]);
            pid += 3;
            updated_x[d].resize(n_velocity+1);
            updated_y[d].resize(n_velocity+1);
            updated_y[d][0] = ystart;
            updated_x[d][0] = xstart;
            for (int i=0;i<n_velocity;i++){
                xstart += mags[i]*cos(thetas[i]+gbias);
                ystart += mags[i]*sin(thetas[i]+gbias);
                updated_x[d][i+1] = xstart;
                updated_y[d][i+1] = ystart;
            }
        }
        size_t k = 0;
        for (size_t i=0;i<(*day_pairs).size();i++){
            int day0 = (*day_pairs)[i].day0;
            int day1 = (*day_pairs)[i].day1;
            vector<Corres>& corres_data = (*day_pairs)[i].corres_data;
            for (size_t j=0;j<corres_data.size();j++){
                int id0=corres_data[j].id0;
                int id1=corres_data[j].id1;
                // if (id0>=updated_x[day0].size() || id1>=updated_x[day1].size()){
                //     cout<<"out of size"<<endl;
                // }
                residuals[k] = updated_x[day0][id0] - updated_x[day1][id1];
                residuals[k+total_num_corres] = updated_y[day0][id0] - updated_y[day1][id1];
                k++;
            }
        }
    	
    	return true;
    }
    static CorresCostFunction* Create(size_t total_num_corres,
                                    vector<Ronin>* all_ronins,
                                    vector<DayPair>* day_pairs,
    	                            vector<double*>* parameter_blocks) {
    	CorresConstraint* constraint = new CorresConstraint(total_num_corres, all_ronins, day_pairs);
    	CorresCostFunction* cost_function = new CorresCostFunction(constraint);
    	// delete this new memory?
    	parameter_blocks->clear();
        for (size_t i=0; i<(*all_ronins).size();i++){
            double* gtransfroms = (*all_ronins)[i].get_gtransforms();
            for(int t=0;t<3;t++){
                parameter_blocks->push_back(&(gtransfroms[t]));
                cost_function->AddParameterBlock(1);
            }
        }
        
    	cost_function->SetNumResiduals(total_num_corres*2);
    	return (cost_function);

    }
    ///////
    vector<Ronin>* all_ronins;
    vector<DayPair>* day_pairs;
    size_t total_num_corres;
};

struct BiasRegConstraint {
	template <typename T>
	bool operator()(const T* const b1, const T* const b2, T* residual) const {
		//residual[0] = min(ceres::abs(b1[0]-b2[0]), T(2.*M_PI)-ceres::abs(b1[0]-b2[0]));
        residual[0] = b1[0]-b2[0];
		return true;
	}
};

bool read_corres_data(string& filename, vector<Corres>& corres_pairs){
    ifstream fin;
    fin.open(filename);
    if (!fin){
        cerr<<"cannot read mag theta"<<endl;
        exit(1);
    }
    string tmp;
    int n_corres;
    fin>>tmp>>n_corres;
    corres_pairs.resize(n_corres);
    for (int i=0; i<n_corres; i++){
        double t1,t2,t3;
        fin>>t1>>t2>>t3;
        //cout<<t1<<" "<<t2<<" "<<t3<<endl;
        corres_pairs[i].id0 = t1;
        corres_pairs[i].id1 = t2;
        corres_pairs[i].dist = t3;
        //cout<<corres_pairs[i].id1<<"\t"<<corres_pairs[i].id2<<"\t"<<corres_pairs[i].dist<<endl;
    }
    fin.close();
    double t_sum = 0.;
    for (int i=0; i<n_corres; i++){
        corres_pairs[i].dist = std::exp(-corres_pairs[i].dist/50.);
        t_sum += corres_pairs[i].dist;
    }
    for (int i=0; i<n_corres; i++){
        corres_pairs[i].dist /= t_sum;
    }
    
    // cout<<n_corres<<endl;
    // for (int i=0; i<n_corres; i++){
    //     cout<<corres_pairs[i].id1<<"\t"<<corres_pairs[i].id2<<"\t"<<corres_pairs[i].dist<<endl;
    // }
    
    return true;
}

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);

    string base_path = "./outputs/";
    ifstream fin(base_path+"folder_list_manual.txt");
    vector<string> folder_ids;
    while(!fin.eof()){
        string tmp;
        fin>>tmp;
        if (tmp==""||tmp=="\n")
            continue;
        folder_ids.push_back(tmp);
    }
    fin.close();
    vector<Ronin> all_ronins;
    all_ronins.resize(folder_ids.size());
    for (size_t i=0;i<folder_ids.size();i++){
        string txtname = base_path+folder_ids[i] + "/c_single_corres_align_incremental.txt";
        cout<<txtname<<endl;
        all_ronins[i].read_mag_theta(txtname);
    }
    cout<<"finish reading "<<all_ronins.size()<<" trajectories"<<endl;
    vector<DayPair> day_pairs;
    size_t total_num_corres = 0;
    for (size_t i=0;i<folder_ids.size();i++){
        for (size_t j=0;j<folder_ids.size();j++){
            if (i>=j)
                continue;
            string txtname = "./experiments/paircorres/"+folder_ids[i]+"-"+folder_ids[j]+".txt";
            DayPair day_pair;
            day_pair.day0 = i;
            day_pair.day1 = j;
            read_corres_data(txtname, day_pair.corres_data);
            day_pairs.push_back(day_pair);
            total_num_corres += day_pair.corres_data.size();
        }
    }
    cout<<"finish reading "<<total_num_corres<<" correspondences"<<endl;
    Problem problem;
    vector<double*> parameter_blocks;
    CorresConstraint::CorresCostFunction* corres_cost_function = 
        CorresConstraint::Create(total_num_corres, &all_ronins, &day_pairs, &parameter_blocks);
    problem.AddResidualBlock(corres_cost_function, new CauchyLoss(0.5), parameter_blocks);

    ceres::Solver::Options options;
    //options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    cout << summary.FullReport() << "\n";

    for (size_t i=0;i<folder_ids.size();i++){
        string txtname = base_path+folder_ids[i] + "/c_multi_corres_align.txt";
        all_ronins[i].write_mag_theta(txtname);
    }

    cout<<"success!"<<endl;
    return 0;
}
