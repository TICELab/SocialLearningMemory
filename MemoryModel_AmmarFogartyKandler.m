% Ammar, M.P., Fogarty, L., Kandler, A.
% Simulation model of cultural evolution including memory:
% 1.  Perception of environment and trait choice
% 2.  Decision on next step: 
%     either
% 2a. Performance of chosen trait
%     or
% 2b. Addition of trait through social learning or individual learning to repertoire and performance of new trait
% 3.  Forgetting
% 4.  Birth - death dynamic

   
clear all
clc

rng('shuffle')

N = 200; % number of individuals
T = 500; % maximum size of cultural repertoire (of a single individual)
nGen = 500*N; % number of time steps
nSim = 1; %number of simulation repeats
pChange = 0.2; % probability of an environmental change (per time step)  
% considered values of pChange: [0 0.0001 0.001 0.005 0.0075 0.01 0.05 0.1 0.2] 

pPerform = 0.8; %learning rate
tau = 0.1; %Softmax parameter
mu_f = 0.05; % probability of mutating propensity of forgetting (per birth event)
mu_s = 0.05; % probability of mutating propensity of social learning (per birth event)
sd = 0.1; % standard deviation of mutation            
verticalError = 0; % 1- fraction of the parent's repertoire that is vertically tranmitted
% considered values of verticalError: [0, 1]
forgettingFilter = 1; % 2: forgetting dynamis is based on trait's benefit; 1: forgetting dynamic is based on number of production events for each trait, 0: forgetting dynamic is random
%considered options: [2,1,0]

%%

for s = 1:nSim
    
    %% Initialisation
    Forget = ones(1,N).*0.2;
    Social = ones(1,N).*0.2;

    Fitness = ones(1,N);
    Repertoire = zeros(N,T,2);
    R_initial = randi(2,[1 N]);
    Age = zeros(1,N);    

    Env_vec = [1 2]; % environmental states
    Env = 1; % start simulations from state 1

    % initialisation of individual repertoires
    for i = 1:N
        Repertoire(i,1,R_initial(i)) = rand; % randomly choose the environment, 1 or 2, to which individuals have a well-adapted trait
        Repertoire(i,1,setdiff(Env_vec,R_initial(i))) = 10^-6*rand; % fitness value of the "other" environment
    end

    ChosenTrait = Repertoire(:,1,1:2); 
    Counts = zeros(N,T); % count record of individual's traits
    Counts(:,1) = 1; % initial training of individual's traits is set to 1

    for t = 1:nGen % loop over time steps

        if rand < pChange
            Env = setdiff(Env_vec,Env); %if environment changes, it switches to alternative state
        end
        
        Fitness_past = Fitness;
        ChosenTrait_past = ChosenTrait;

        for i = 1:N % loop over all individuals

            %% Recall and trait choice
            AvailableTraits = Repertoire(i,:,Env); % recall adaptive values of all traits previously produced in the current state        
            
            if sum(AvailableTraits)>0 % if cultural repertoire isn't empty, choose a trait
                
                choice = randsample(T,1,true,AvailableTraits); % probabilistic choose of a trait from repertoire weighted by adapative value
                %%
                % considered options: softmax function of trait choice
                
                %%
                % * a = exp(AvailableTraits/tau)/sum(exp(AvailableTraits/tau)); 
                % * choice = randsample(T,1,true,a); %traits chance of being chosen is exponentially proportional to benefit 
                ChosenTrait(i,:) = Repertoire(i,choice,1:2); 
                LearnProbability = 1-ChosenTrait(i,Env); % adaptive value of chosen trait determines the probabiliy of learning

            else % if cultural repertoire is empty, learn with probability 1
                
                LearnProbability = 1;
                
            end

            %% Learning   
            repFull = numel(find(sum(Repertoire(i,:,:),3)==0));
            if rand>LearnProbability | repFull == 0 % Depending on the expected benefit of the chosen trait or if repertoire is full...
            %%
            % considered options: constant rate of learning

            %%
            % if rand<pPerform| repFull == 0 
            
                Fitness(i) = ChosenTrait(i,Env); % ... perform the chosen trait
                Counts(i,choice) = Counts(i,choice)+1; % update count record 
 
            else % otherwise learn a new trait (and add it to the repertoire)

                if rand<Social(i) % depending on intrinsic propensity, learn socially
                    
                    RoleModel = randsample(N,1,true,Fitness_past); % choose a role model according to observed fitness levels in the last round                                          
                    %%
                    % considered options: unbiased social learning
                    
                    %%
                    % * RoleModel= randi(N);
                    LearnedTrait = ChosenTrait_past(RoleModel,:); % 'observe' adaptive value of role model's produced trait 
                    HaveIt = find(Repertoire(i,:,1)==LearnedTrait(1)); % check whether 'observed; trait is already in individual i's repertoire (based on the adaptive value in the state to which it is adapted)

                    if numel(HaveIt)>=1 % if 'observed'  trait already in repertoire

                        choice = find(Repertoire(i,:,1)==LearnedTrait(1)); % find index of 'observed' trait in individual i's repertoire
                        
                        if numel(choice) > 1 % check whether 'observed' trait exists multiple times (due to random innovation)
                            choice = randsample(numel(choice),1); % choose one
                        end
                        
                        Fitness(i) = LearnedTrait(Env); % ... and perform this trait
                        Counts(i,choice) = Counts(i,choice)+1; % update count record

                    else % if 'observed'  trait is not in repertoire 

                        spot = find(sum(Repertoire(i,:,1:2),3)==0,1,'first'); % find empty spots in one's cultural repertoire and choose the first one                                              
                        Repertoire(i,spot,1:2) = LearnedTrait; % add adaptive value of the learnt trait
                        Fitness(i) = LearnedTrait(Env); % perform the trait 
                        Counts(i,spot) = Counts(i,spot)+1; % update count record

                    end

                    ChosenTrait(i,:) = LearnedTrait;

                else %...innovate a trait

                    Envi = Env; % clever innovation
                    %%
                    % considered options: random innovation 
                    
                    %%
                    % * Envi = randi(2);                     
                    spot = find(sum(Repertoire(i,:,1:2),3)==0,1,'first'); % find empty spots in one's cultural repertoire and choose the first one                    
                    aMax_1 = rand; 
                    aMax_2 = 10^-6*rand;
                    Repertoire(i,spot,Envi) = aMax_1; % add adaptive value for current state
                    Repertoire(i,spot,setdiff(Env_vec,Envi)) = aMax_2; % add adaptive value for alternative state
                    Fitness(i) = Repertoire(i,spot,Env); % perform the trait
                    Counts(i,spot) = Counts(i,spot)+1; % update count record
                    ChosenTrait(i,:) = Repertoire(i,spot,1:2);

                end
                
            end

            %% Forgetting 
            % Given an intrinsic rate of forgetting, erase a trait from repertoire other than the currently performed based on frequency of trait's usage
            Did = find(Repertoire(i,:,Env)==Fitness(i)); % Remember currently performed trait
            AvailableTraits = find(sum(Repertoire(i,:,1:2),3)~=0); % indices of all traits
            
            if rand<Forget(i) && numel(AvailableTraits)>0 % inididual i is prone to forgetting and cultural repertoire is not empty

                AvailableTraits(AvailableTraits==Did)=[];                  

                if forgettingFilter==1 && numel(AvailableTraits) > 1 % forgetting dynamic based on count record --- least used traits are forgotten with higher probability 
                    
                    Counts_h = Counts(i,AvailableTraits); % count record of all traits but the currently performed one
                    ForgetThis = randsample(AvailableTraits,1,'true',(1-(Counts_h)/sum(Counts_h)));  % choose trait to forget based on count record

                elseif forgettingFilter==0 && numel(AvailableTraits) > 1  % random forgetting dynamic 

                    ForgetThis = randsample(AvailableTraits,1); % choose a random trait to forget

                elseif forgettingFilter==2 && numel(AvailableTraits) > 1  % forgetting dynamic based on trait's adaptive value - least beneficial is forgotten with higher probability
                    
                    Benefit_h = sum(Repertoire(i,AvailableTraits,1:2),3); %sum benefit of all traits across both environments
                    ForgetThis = randsample(AvailableTraits,1,'true',(1-(Benefit_h)/sum(Benefit_h)));  % choose trait to forget based on adaptive value
                
                elseif numel(AvailableTraits) == 1

                    ForgetThis = AvailableTraits;
                
                elseif numel(AvailableTraits) == 0                        
		    
                    ForgetThis = [];                    

                end

                Counts(i,ForgetThis) = 0; % reset count of trait production                 
                Repertoire(i,ForgetThis,:) = 0; % erase trait from repertoire
                                                
                %%
                % considered option: a single production is forgotten from trait chosen based on frequency of usage
                %%
                % * Counts(i,ForgetThis) = Counts(i,ForgetThis)-1; %  
                %%
                % * if Counts(i,ForgetThis)<=0 % if there is no past production, erase trait from repertoire
                %%
                % * Repertoire(i,ForgetThis,:) = 0;
                %%
                % * end
                
            end

        end

        %% Turn over
        parent = randsample(N,1,true,Fitness); % reproducing individual is chosen based on current performances
        if t == 1
            dead = randi(N,1); % in the very first time step of each simulation the dying individual is chosen at random
        else
            dead = randsample(N,1,true,Age); % individual to die is chosen based on age
        end   
        
        Age(1,setdiff(1:N,dead))=Age(1,setdiff(1:N,dead))+1;
        Age(1,dead) = 0;        
        
        % spot of "dead" individual becomes re-occupied by naive individual inheriting its parent's properties
        Forget(dead) = Forget(parent);
        Social(dead) = Social(parent);

        if rand<mu_f % mutation of propensity to forget
            Forget(dead) = Forget(dead)+sd*randn(1);
            
            if Forget(dead)>1
                
                Forget(dead)=1;
                
            elseif Forget(dead)<0
                
                Forget(dead) = 0;
                
            end
            
        end

        if rand<mu_s % mutation of propensity to socially learn
            Social(dead) = Social(dead)+sd*randn(1);
            
            if Social(dead)>1
                
                Social(dead)=1;
            
            elseif Social(dead)<0
                
                Social(dead) = 0;
            
            end
            
        end

        % vertical transmission of traits 
        parentTraits = find(sum(Repertoire(parent,:,1:2),3)>0); % find all available traits of parents 
        x = binornd(numel(parentTraits),verticalError); % number of traits to be transmitted 
        offspringTraits = parentTraits(randsample(1:numel(parentTraits),numel(parentTraits)-x)); % randomly choose parentTraits-x traits from parent's repertoire 

        Repertoire(dead,:,:) = 0; % empty cultural repertoire for naive individual        
        Counts(dead,:) = 0; % reset the count record for naive individual        
                
        for tr = 1:numel(offspringTraits)
            
            Repertoire(dead,tr,1) = Repertoire(parent,offspringTraits(tr),1); % add adaptive value of all traits to state 1
            Repertoire(dead,tr,2) = Repertoire(parent,offspringTraits(tr),2); % ... and state 2
            Counts(dead,tr) = 1; % set count record to 1

        end    
        
    end

end

