/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

// declare a random engine to be used across multiple and various method call
static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	default_random_engine gen;

	num_particles = 100;

	// define normal distribution for sensor noise
	normal_distribution<double> x_init_noise(0, std[0]);
	normal_distribution<double> y_init_noise(0, std[1]);
	normal_distribution<double> theta_init_noise(0, std[2]);

	// init particle
	for(unsigned int i = 0; i < num_particles; i++){
		Particle p;
		p.id = i;
		p.x = x;
		p.y = y;
		p.theta = theta;
		p.weight = 1.0;

		// add noise
		p.x += x_init_noise(gen);
		p.y += y_init_noise(gen);
		p.theta += theta_init_noise(gen);

		// add particle into set
		particles.push_back(p);
		weights.push_back(p.weight);
	}

	is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	// define normal distributions for sensor noise
	normal_distribution<double> dist_x(0,std_pos[0]);
	normal_distribution<double> dist_y(0,std_pos[1]);
	normal_distribution<double> dist_theta(0,std_pos[2]);
	default_random_engine gen;

	for(unsigned int i = 0; i < num_particles; i++){

		// judge if yaw_rate is too small(almost 0)
		if(fabs(yaw_rate) < 0.00001){
			particles[i].x += velocity * delta_t * cos(particles[i].theta);
			particles[i].y += velocity * delta_t * sin(particles[i].theta);
		}
		else{
			particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
			particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
			particles[i].theta += yaw_rate * delta_t;
		}
		// add noise 
		particles[i].x += dist_x(gen);
		particles[i].y += dist_y(gen);
		particles[i].theta += dist_theta(gen);
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	for(unsigned int i = 0; i < observations.size(); i++){

		// current observation
		LandmarkObs current_observation = observations[i];

		// initialize min_dist
		double min_dist = numeric_limits<double>::max();

		int map_id = -1;

		for(unsigned int j = 0; j < predicted.size(); j++){

			// current prediction
			LandmarkObs current_prediction = predicted[j];
			double distance = dist(current_prediction.x,current_prediction.y,current_observation.x,current_observation.y);

			if(distance < min_dist){
				min_dist = distance;
				map_id = current_prediction.id;
			}

		}

		// set the observation's id to nearest predicted landmark's id
		observations[i].id = map_id;

	}

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	// compute the left of Multivariate Normal Distribution
	const double left = 1 / (2 * M_PI * std_landmark[0] * std_landmark[1]);

	// compute the x and y denom
	const double x_denom = 2 * std_landmark[0] * std_landmark[0];
	const double y_denom = 2 * std_landmark[1] * std_landmark[1];

	for(unsigned int i = 0; i < num_particles; i++){

		// get info from each particle
		double cur_x = particles[i].x;
		double cur_y = particles[i].y;
		double cur_theta = particles[i].theta;

		// create a vector to hold the landmarks in range
		vector<LandmarkObs> range_landmarks;

		// get landmark lists
		// add the prediction that the sensor can capture
		vector<Map::single_landmark_s> landmark_list = map_landmarks.landmark_list;
		for(unsigned int j = 0; j < landmark_list.size(); j++){
			
			double lm_x = landmark_list[j].x_f;
			double lm_y = landmark_list[j].y_f;
			int lm_id = landmark_list[j].id_i;

			double cir_dist = dist(lm_x,lm_y,cur_x,cur_y);

			if(cir_dist <= sensor_range){
				range_landmarks.push_back(LandmarkObs{ lm_id, lm_x, lm_y });
			}

		}

		

		vector<int> associations;
		vector<double> sense_x;
		vector<double> sense_y;

		// init weights
		particles[i].weight = 1.0;
		weights[i] = 1.0;


		// transform coordinate system
		std::vector<LandmarkObs> transform_obs;
		for(unsigned int k = 0; k < observations.size(); k++){

			LandmarkObs best_landmark;

			double t_x = cur_x + cos(cur_theta) * observations[k].x - sin(cur_theta) * observations[k].y;
			double t_y = cur_y + sin(cur_theta) * observations[k].x + cos(cur_theta) * observations[k].y;
			transform_obs.push_back(LandmarkObs{ observations[k].id, t_x, t_y });

			double dist_min = numeric_limits<double>::max();
			bool bestfound = false;

			// for current observation, check every landmark in sensor range find the best landmark
			for(unsigned int n = 0; n < range_landmarks.size(); n++) {
				double distance = dist(range_landmarks[n].x,range_landmarks[n].y,t_x,t_y);

				if(distance < dist_min){
					dist_min = distance;
					best_landmark = range_landmarks[n];
					bestfound = true;
				}
			}

			if(bestfound){
				double x_diff = t_x - best_landmark.x;
				double y_diff = t_y - best_landmark.y;

				double right = (x_diff * x_diff) / x_denom + (y_diff * y_diff) / y_denom;
				// update weights
				particles[i].weight *= (left * exp(-right));
				weights[i] = particles[i].weight;
			}

			// add detail to association, sense_x and sense_y
			associations.push_back(best_landmark.id);
			sense_x.push_back(t_x);
			sense_y.push_back(t_y);
		}
		//cout << ".." << endl;
		// perform dataAssociation for the predictions and transformed observations on current particle.
		//dataAssociation(predictions, transform_obs);

		SetAssociations(particles[i],associations,sense_x,sense_y);
	}

}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	default_random_engine gen;

	// prepare for new particles
	vector<Particle> new_particles;

	// create discrete distribution for weights
	discrete_distribution<int> r_index(weights.begin(),weights.end());

	// spin the resample wheel
	for(unsigned int j = 0; j < num_particles; j++){
		int idx = r_index(gen);
		new_particles.push_back(particles[idx]);
	}

	particles = new_particles;

}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

    return particle;
}	

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
