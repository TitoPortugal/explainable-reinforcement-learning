import argparse
import gym
import numpy as np
import os
from os import path
from collections import deque
from CarRacingDQNAgent import CarRacingDQNAgent
from common_functions import process_state_image
from common_functions import generate_state_frame_stack_from_queue

# To monitorize CPU and Memory
import psutil

RENDER                        = True
STARTING_EPISODE              = 1
ENDING_EPISODE                = 1000
SKIP_FRAMES                   = 2
TRAINING_BATCH_SIZE           = 64
SAVE_TRAINING_FREQUENCY       = 20
UPDATE_TARGET_MODEL_FREQUENCY = 5

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training a DQN agent to play CarRacing.')
    parser.add_argument('-m', '--model', help='Specify the last trained model path if you want to continue training after it.')
    parser.add_argument('-lm','--learningmodel',help='Specify the last trained learning model path if you want to continue training after it.')
    parser.add_argument('-s', '--start', type=int, help='The starting episode, default to 1.')
    parser.add_argument('-e', '--end', type=int, help='The ending episode, default to 1000.')
    parser.add_argument('-p', '--epsilon', type=float, default=1.0, help='The starting epsilon of the agent, default to 1.0.')
    args = parser.parse_args()
    
    pid = os.getpid()
    process = psutil.Process(pid)
    process.cpu_percent(interval=1)
    # print("n cpus: ", psutil.cpu_count())
    # print("Memory info: ", process.memory_info())
    # print("Cpu percent: ", process.cpu_percent(interval=1))
    
    env = gym.make('CarRacing-v0')
    agent = CarRacingDQNAgent(epsilon=args.epsilon)
    if args.model:
        agent.load(args.model)
    if args.learningmodel:
        agent.load_learning_model(args.learningmodel)
    if args.start:
        STARTING_EPISODE = args.start
    if args.end:
        ENDING_EPISODE = args.end
    
    # rewards = []
    results_folder = 'results/agent11/'

    for e in range(STARTING_EPISODE, ENDING_EPISODE+1):
        init_state = env.reset()
        init_state = process_state_image(init_state)

        total_reward = 0
        negative_reward_counter = 0
        state_frame_stack_queue = deque([init_state]*agent.frame_stack_num, maxlen=agent.frame_stack_num)
        time_frame_counter = 1
        done = False
        
        # q_values = []
        # probabilities_learning = []
        probabilities_introspection = []
        
        
        while True:
            if RENDER:
                env.render()

            current_state_frame_stack = generate_state_frame_stack_from_queue(state_frame_stack_queue)
            action = agent.act(current_state_frame_stack)
            # print("current_state_frame_stack: " + str(current_state_frame_stack))
            

            reward = 0
            for _ in range(SKIP_FRAMES+1):
                next_state, r, done, info = env.step(action)
                reward += r
                if done:
                    break

            # If continually getting negative reward 10 times after the tolerance steps, terminate this episode
            negative_reward_counter = negative_reward_counter + 1 if time_frame_counter > 100 and reward < 0 else 0

            # Extra bonus for the model if it uses full gas
            if action[1] == 1 and action[2] == 0:
                reward *= 1.5

            total_reward += reward
            # print(total_reward)

            next_state = process_state_image(next_state)
            state_frame_stack_queue.append(next_state)
            next_state_frame_stack = generate_state_frame_stack_from_queue(state_frame_stack_queue)

            agent.memorize(current_state_frame_stack, action, reward, next_state_frame_stack, done)
            
            #store predicted q values 
            # q_values.append(agent.model.predict(np.expand_dims(current_state_frame_stack, axis=0))[0])
            # store probability of success
            # probabilities_learning.append(agent.learning_model.predict(np.expand_dims(current_state_frame_stack, axis=0))[0])
            probabilities_introspection.append(agent.introspection_probabilitysucces(current_state_frame_stack, total_reward))
            

            if done or negative_reward_counter >= 25 or total_reward < 0:
                print('Episode: {}/{}, Scores(Time Frames): {}, Total Rewards(adjusted): {:.2}, Epsilon: {:.2}'.format(e, ENDING_EPISODE, time_frame_counter, float(total_reward), float(agent.epsilon)))
                # print('Probability Success (learning): ' + str(agent.learning_model.predict(np.expand_dims(current_state_frame_stack, axis=0))))
                # print('Probability Success (learning): ' + str(agent.target_learning_model.predict(np.expand_dims(current_state_frame_stack, axis=0))))
                # print('Probability Success (introspection): ' + str(agent.introspection_probabilitysucces(current_state_frame_stack, total_reward)))
                
                break
            if len(agent.memory) > TRAINING_BATCH_SIZE:
                agent.replay(TRAINING_BATCH_SIZE)
                # agent.replay_learning(TRAINING_BATCH_SIZE)
            time_frame_counter += 1

        if e % UPDATE_TARGET_MODEL_FREQUENCY == 0:
            agent.update_target_model()
            # agent.update_target_learning_model()

        if e % SAVE_TRAINING_FREQUENCY == 0:
            agent.save('./save/agent11/model/trial_{}.h5'.format(e))
            # agent.save_learning_model('./save/agent9/learning/trial_{}.h5'.format(e))
                    
        # store rewards
        # rewards.append(float(total_reward))
        
        # episode_folder = 'episode' + str(e) + '/'
        # if not path.exists(results_folder + episode_folder):
        #     os.mkdir(results_folder + episode_folder)
        
        # probabilities_learning = np.nan_to_num(probabilities_learning)
        # probabilities_introspection = np.nan_to_num(probabilities_introspection)
        # np.savetxt(results_folder + episode_folder + 'Qvalues.csv',np.array(q_values),delimiter=",")
        # np.savetxt(results_folder + episode_folder + 'ProbabilitiesLearning.csv',np.array(probabilities_learning),delimiter=",")
        # np.savetxt(results_folder + episode_folder + 'ProbabilitiesIntrospection.csv',np.array(probabilities_introspection),delimiter=",")
        
    # with open(results_folder + 'Reward.csv','ab') as f:
    #     np.savetxt(f, np.array(rewards),delimiter=",")
    #     # f.write(b"\n")
    
    cpu_percent = process.cpu_percent(interval=None)
    cpu_percent /= psutil.cpu_count()
    mem_info = process.memory_info()
    with open(results_folder + 'Memory.csv','ab') as f:
        # f.write(str(mem_info.vms/(2**20)))
        np.savetxt(f, np.array([mem_info.vms/(2**20)]),delimiter=",") #Store vms in MB
    
    with open(results_folder + 'Cpu.csv','ab') as f:
        # f.write(str(cpu_percent))
        np.savetxt(f, np.array([cpu_percent]),delimiter=",") 


    env.close()
