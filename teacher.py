import numpy as np


# parameters for the pure pursuit controller
POSITION_THRESHOLD = 0.04
REF_VELOCITY = 0.5
GAIN = 10
FOLLOWING_DISTANCE = 0.3


class PurePursuitExpert:
    def __init__(self, env, ref_velocity=REF_VELOCITY, position_threshold=POSITION_THRESHOLD,
                 following_distance=FOLLOWING_DISTANCE, max_iterations=1000):
        self.env = env
        self.following_distance = following_distance
        self.max_iterations = max_iterations
        self.ref_velocity = ref_velocity
        self.position_threshold = position_threshold
        self.running_predictions = np.zeros(5)

    def predict(self, num_step, observation):  # we don't really care about the observation for this implementation
        closest_point, closest_tangent = self.env.closest_curve_point(self.env.cur_pos, self.env.cur_angle)
        #print(closest_point, closest_tangent)

        iterations = 0
        lookup_distance = self.following_distance
        curve_point = None

        while iterations < self.max_iterations:
            # Project a point ahead along the curve tangent,
            # then find the closest point to that
            if closest_tangent is None or lookup_distance is None:
                print("Detected anomaly:", closest_tangent, lookup_distance)
                break

            follow_point = closest_point + closest_tangent * lookup_distance

            curve_point, _ = self.env.closest_curve_point(follow_point, self.env.cur_angle)

            # If we have a valid point on the curve, stop
            if curve_point is not None:
                break

            iterations += 1
            lookup_distance *= 0.5

        if curve_point is None:# invalid action
            return None

        # Compute a normalized vector to the curve point
        point_vec = curve_point - self.env.cur_pos
        point_vec /= np.linalg.norm(point_vec)

        dot = np.dot(self.env.get_right_vec(), point_vec)
        steering = GAIN * -dot
        self.running_predictions[num_step % self.running_predictions.shape[0]] = steering
        steering = self.running_predictions.mean()
        #print("Prediction {} in {} iterations:".format(steering, iterations))

        if np.abs(steering) > 0.8:
            velocity = self.ref_velocity * 0.5
        else:
            velocity = self.ref_velocity

        return velocity, steering

