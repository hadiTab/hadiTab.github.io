---
layout: post
title:  "The Kalman Filter"
date:   2017-10-24 -0600
categories: robotics
mathjax: true
---
{% include mathjax.html %}

Kalman filters are recursive state estimators for linear Gaussian systems. Given one, or multiple observations, a Kalman filter provides an estimate of the state of the system. The output is a least squares optimal estimate based on the observations and an internal model that continuously predicts the next state. Each input value should be associated with an estimate of it’s uncertainty, expressed as a covariance. The state estimation model will also have an associated uncertainty, often referred to as process noise. A neat property of the Kalman filter is that since the output is an optimal estimate.


Systems and environments are characterized by **state**. A state is a set of variables that have an impact on the future of the system or environment. There is no *one* correct set of state variables; however, if the state variables fail to sufficiently describe the system we will have difficulty estimating future states. A robot moving around on the ground can be considered to be constrained to a plane, and it’s state could merely consist of the planar coordinates of the robot. One could argue that the direction the robot is facing, or its *heading* described by an angular offset from a predefined direction (i.e. the magnetic north) , will also affect the future states and should be included as a state variable. The selection of the state variables can have a significant effect on the accuracy of the estimated states.


Systems and environments are characterized by **state**. A state is a set of variables that have an impact on the future of the system or environment. There is no *one* correct set of state variables; however, if the state variables fail to sufficiently describe the system we will have difficulty estimating future states. A robot moving around on the ground can be considered to be constrained to a plane, and it’s state could merely consist of the planar coordinates of the robot. One could argue that the direction the robot is facing, or its *heading* described by an angular offset from a predefined direction (i.e. the magnetic north) , will also affect the future states and should be included as a state variable. The selection of the state variables can have a significant effect on the accuracy of the estimated states.


The state is represented by a vector containing all of the state variables. For the example of a robot on a planar surface the state vector $X$ at a time $t$ could be:

$$
X_t = \begin{bmatrix} x_t \\ y_t \\ \theta_t \end{bmatrix} 
$$

If velocities were to be considered, the state vector could be written as:

$$
X_t = \begin{bmatrix} x_t \\ y_t \\ \theta_t \\ \dot{x}_t \\ \dot{y}_t \\ \dot{\theta}_t \end{bmatrix} 
$$

The number of elements in the state vector directly influence the computational cost of running a Kalman filter, because of this Kalman filters are often described by the number of elements in their state vector. For example, a Kalman filter designed for the above state vector would be referred to as a 6-state Kalman filter.


As previously mentioned, the Kalman filter will predict the future states. A **state transition model** predicts the future state by performing a linear transformation on the state vector:

$$
X_{t}^- = A_tX_{t-1} + w 
$$


$A_t$ is an $n \times n$ matrix, where $n$ is the dimension of the state vector $X_t$, and $w$ is additive noise. The superscript $^-$ indicates that this is the predicted state prior to the integration of any observations. This state is referred to as the **prior** state. In some cases we may have access to control variables, i.e. commanded velocities for a mobile robot. In such cases, the control variables can also be included in the state transition model:

$$
X_{t}^- = A_tX_{t-1} + B_tu_t + w 
$$

Here, $u_t$ is an $m$-element vector describing the control variables, and $B_t$ is an $n \times m$ matrix that transforms $u$ into an $n$-dimensional vector compatible with the state vector.


Let’s continue the discussion with a simple example of the one dimensional position of a robot (i.e. a robot that is constrained to a linear rail). The state variable for our filter will be the position along the single axis represented by $x$. A sensor measures the linear position and outputs an estimate $z$ at some frequency with some uncertainty modeled as white noise. The figure below shows the measured position of the robot over a 100 time-step interval.

![image1](/images/kf_01.png)
<div><p class="wp-caption-text">The robot position along the x-axis. There is a good amount of noise.</p></div>

These measurements are somewhat noisy and we probably want a smoother position output since this output will likely be used for path planning and motion control. Let’s try getting a smoother output by using a Kalman filter. We need to come up with a state prediction model for the internal prediction step of the filter. The model predicts the future state of the robot based on the present state. Taking another look at the figure above, you may notice that changes in position usually happen at a constant rate. Meaning that the robot usually travels at a constant velocity. Let’s make this our state transition model. This is what the state transition model would look like:

$$
\begin{bmatrix} x_t^- \\ v_t^- \end{bmatrix} = \begin{bmatrix} 1 & \delta t \\ 0 & 1 \end{bmatrix}\begin{bmatrix} x_{t-1} \\ v_{t-1} \end{bmatrix} 
$$

Note that a velocity $v$ has been added to the state vector, $\delta t$ is the time-step, and $P$ is the process noise. The process noise, is our way of introducing uncertainty into the prediction state. The process noise is also modeled as white noise, therefore it will only directly affect the covariance of the predicted state.
In a similar manner, sing the same model, we predict the covariance of the future state:

$$
\Sigma_{t}^- = A \Sigma_{t-1} A^\top + \Sigma_w 
$$

where $\Sigma$ indicates a covariance matrix. This concludes the prediction stage of the Kalman filter. Note that we have not included any control variables; for simplicity we will assume that control variables are not accessible.


The next step in the Kalman filter is to update the estimated state by integrating observations. This operation will yield the **posterior** state, which is the output of the filter at each time interval.
The observations are introduced by adding in the weighted difference of the predicted state and the observed state:

$$
X_t = X_t^- + K_t (z_k – H_tX_t^-) 
$$

$K_t$ is the Kalman gain, which determines the weight of the observations. The Kalman gain is calculated at each time-step, as will be shown later. $H_t$, sometimes referred to as the extraction matrix, transforms the state vector into a vector compatible with the observation vector. In this case, it would simply be:

$$
H = \begin{bmatrix} 1 & 0 \end{bmatrix} 
$$

In a similar manner, the state covariance would be:

$$
\Sigma_t = (I – K_t H)\Sigma_t^- 
$$

The Kalman gain is actually where all the magic happens and is calculated as:

$$
K_t = \Sigma_t^- H^\top (H \Sigma_t^- H^\top + \Sigma_z)^{-1} 
$$

Note that the Kalman gain only depends on the covariance matrices and not the mean values. This is an indication that the weighting of the variables is based on their uncertainty and not their absolute value.


We have now completed a full prediction and update cycle. In practice, we would have the filter going through this cycle at our desired output rate. If the desired rate was to be higher than the measurement frequency, we would simply skip the update step until the next observations became available.


Going back to our example data, by applying a Kalman filter we get much smoother position estimates for the robot as seen below:


![image2](/images/kf_02.png)
<p class="wp-caption-text">The Kalman filter provides smooth estimates but overshoots at sudden velocity changes.</p></div>


The filter definitely smoothened out the observations, however it does overshoot with sudden velocity changes. This is expected because of the constant velocity assumption in the state transition model. The quality of the output is also highly dependent on the process noise. For instance, increasing the uncertainty in the velocity component of the state vector compared to the position component mitigates the overshoots to some extent. I’ve included a copy of the python code I used to generate data and for the Kalman filter used in this example. You can try playing around with the tuning variables to get a better understanding of how they affect the results. Also, note that this is a ‘quick and dirty’ implementation of a Kalman filter for this example only, I would not suggest using it in any serious project.

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

class KalmanFilter:
    def __init__(self):
        self.x_prior = np.matrix([[2],[0]])
        self.x_post = np.matrix([[2],[0]])
        self.sigma_prior = np.matrix([[0.25, 0],[0, 0.25]]) 
        self.sigma_post = np.matrix([[0.3, 0],[0, 4]]) 
        self.sigma_P = np.matrix([[0.5, 0],[0, 3]])
        self.dt = 1
	self.obs_var = 2.5
    
    def model(self):
        A = np.matrix([[1, self.dt],[0, 1]])
        self.x_prior = A*self.x_post
        self.sigma_prior = np.transpose(A)*self.sigma_post*A + self.sigma_P
        
    def update(self, z):
        K = self.sigma_prior*np.transpose(H)*np.linalg.inv(H*self.sigma_prior*np.transpose(H) + self.obs_var)
        self.x_post = self.x_prior + K*(z - H*self.x_prior)
        self.sigma_post = (np.matrix([[1, 0],[0, 1]]) - K*H)*self.sigma_prior
        
    def run(self, z):
        self.x_prior = self.x_post
        self.sigma_prior = self.sigma_post
        self.model()
        self.update(z)
        return self.x_post.item(0)

if __name__ == "__main__":
	
	# generating fake observations
	mu, sigma = 0, 0.1 
	s = np.random.normal(mu, sigma, 100)

	z = np.zeros(100)
	for i in range(int(len(z)/2)):
	    z[i] = 0.5*i + 2
	for i in range(int(len(z)/2), int(2*len(z)/3)):
	    z[i] = 27
	for i in range(int(2*len(z)/3), len(z)):
	    z[i] = i*0.1 + 20.5

	z += 4*s
	
	
	kf = KalmanFilter()
	x = []
	for i in range(100):
		x.append(kf.run(z[i]))

	plt.figure(figsize=(15,10))
	matplotlib.rcParams.update({'font.size': 15})
	plt.plot(z, marker='o', c='mediumaquamarine', markeredgecolor='lightseagreen',
		 markeredgewidth=1, linewidth=5, markersize=6, alpha=1, label='Sensor output')
	plt.plot(x,c='black', linewidth=2, alpha=1, label='Kalman Filter output')
	plt.ylabel('Position (x)')
	plt.xlabel('Time step (t)')
	plt.legend(loc='upper left')

	plt.show()
```

#### References:
For a detailed discussion, mathematical derivation, and much much more please refer to [*Probabilistic Robotics*](http://www.probabilistic-robotics.org/) by Sebastian Thrun, Wolfram Burgard, and Dieter Fox.