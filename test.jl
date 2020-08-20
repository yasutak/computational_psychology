# -*- coding: utf-8 -*-
# %% [markdown]
# # Computational Modeling of Behavioral Data by Prof. Kentaro Katahira

# %% [markdown]
# ## Rescorla-Wagner model
#
#
# $$\begin{aligned}
# V_{t + 1} &= V_{t} + \alpha * (r_{t} - V_{t}) \\
# V_t&: strength\ of\ association\ at\ time\ t \\
# \alpha&: learning\ rate \\
# r_t&: reward\ at\ time t \\
# \end{aligned}$$

# %%
using Plots
using Interact
using Random

# %%
"""
Nₜ: number of trials
α: learning rate
Pᵣ: probability of getting reward
seed: random seed
"""
function plot_rescorla_wagner_model(Nₜ, α, Pᵣ, seed)
    
    rng = MersenneTwister(seed) # fix random seed

    𝐕 = zeros(Nₜ) # initialize strengths of association as Nₜ-length vector
    𝐑 = rand(rng, Nₜ) .< Pᵣ # presence of reinforcer (1 or 0) as Nₜ-length vector

    for t in 1:Nₜ-1
        𝐕[t+1] = 𝐕[t] + α * (𝐑[t] - 𝐕[t])
    end

    plot(𝐕, label= string("a ", α))
    plot!([(i, Pᵣ) for i in 1:1:Nₜ], label="expected value of r: " * string(Pᵣ))
    xlabel!("number of trials")
    ylabel!("strength of association")
    ylims!((0, 1))
    title!("Rescorla-Wagner model")
end

# %% [markdown]
# ### Interactive Plot

# %%
@manipulate for Nₜ=0:1:500, α=0:0.05:1, Pᵣ=0:0.05:1, seed=1:100:1000

    plot_rescorla_wagner_model(Nₜ, α, Pᵣ, seed)
end

# %% [markdown]
# ## Q-learning simulation
# ### softmax function
#
# $$\begin{aligned}
# P(a_t = A) &= \frac{\exp({\beta*Q_t(A))}}{\exp({\beta*Q_t(A))} + \exp({\beta*Q_t(B))}} \\
# &= \frac{1}{1 + \exp({-\beta*(Q_t(A) - Q_t(B)))}} \\
# &= \frac{1}{1 + \exp({-\beta*(\Delta Q))}}\  (\Delta Q = (Q_t(A) - Q_t(B)))
# \end{aligned}$$

# %%
function softmax(β, ΔQ)
    return 1 / (1+ exp(-β * (ΔQ)))
end

# %% [markdown]
# ### Interactive plot

# %%
@manipulate for β in 0:0.05:5
    plot([(Δq, softmax(β, Δq)) for Δq in -4:0.1:4], m=:o, label=string("beta ", β))
    xlabel!("difference in Q")
    ylabel!("probability")
    ylims!((0, 1))
    title!("Softmax Function")
end

# %% [markdown]
# ## Define plot function of Q-learning model

# %%
"""
Nₜ: number of trials
α: learning rate
β: inverse temperature
Pᵣ: probability of getting reward in A
seed: random seed
"""
function plot_q_learning_model(Nₜ, α, β, Pᵣ, seed)
    rng = MersenneTwister(seed)

    𝐐 = zeros(Real, (2, Nₜ)) #initial value of Q in 2 by Nₜ matrix
    𝐜 = zeros(Int, Nₜ) #initial choice in each Nₜ trial
    𝐫 = zeros(Nₜ) # 0 (no reward) or 1 (reward) in each Nₜ trial
    Pₐ = zeros(Nₜ) # probability of choosing A in each trial
    P = (Pᵣ, 1-Pᵣ)

    for t in 1:Nₜ-1
        Pₐ = softmax(β, 𝐐[1, t] - 𝐐[2, t])

        if rand(rng) < Pₐ
            𝐜[t] = 1 #choose A
            𝐫[t] = Int(rand(rng) < P[1])
        else
            𝐜[t] = 2 #choose B
            𝐫[t] = Int(rand(rng) < P[2])
        end

        𝐐[𝐜[t], t+1] = 𝐐[𝐜[t], t] + α * (𝐫[t] - 𝐐[𝐜[t], t])
        𝐐[3 - 𝐜[t], t+1] = 𝐐[3 - 𝐜[t], t] # retain value of unpicked choice
    end

    plot(𝐐[1, :], label="Qt(A)", color="orange")
    plot!([(i, P[1]) for i in 1:1:Nₜ], label="expected value of reward for A:" * string(P[1]), color="darkorange")
    plot!(𝐐[2, :], label="Qt(B)", color="skyblue")
    plot!([(i, P[2]) for i in 1:1:Nₜ], label="expected value of reward for B:" * string(P[2]), color="darkblue")
    xlabel!("number of trials")
    ylabel!("Q (value of behavior?)")
    ylims!((0, 1))
    title!("Q-learning model")
end

# %% [markdown]
# ### Interactive plot

# %%
@manipulate for Nₜ in 0:5:200, α in 0:0.05:1, β in 0:0.25:5, Pᵣ in 0:0.05:1, seed = 1:1:1000
    plot_q_learning_model(Nₜ, α, β, Pᵣ, seed)
end

# %% [markdown]
# ## Parameter Estimation of Q-learing model
# ### Preparation

# %%
function generate_qlearning_data(Nₜ, α, β, Pᵣ)

    𝐐 = zeros(Real, (2, Nₜ)) #initial value of Q in 2 by Nₜ matrix
    𝐜 = zeros(Int, Nₜ) #initial choice in each Nₜ trial
    𝐫 = zeros(Int, Nₜ) # 0 (no reward) or 1 (reward) in each Nₜ trial
    Pₐ = zeros(Real, Nₜ) # probability of choosing A in each trial
    P = (Pᵣ, 1-Pᵣ)

    for t in 1:Nₜ-1
        Pₐ = softmax(β, 𝐐[1, t] - 𝐐[2, t])

        if rand() < Pₐ
            𝐜[t] = 1 #choose A
            𝐫[t] = (rand() < P[1])
        else
            𝐜[t] = 2 #choose B
            𝐫[t] = Int(rand() < P[2])
        end

        𝐐[𝐜[t], t+1] = 𝐐[𝐜[t], t] + α * (𝐫[t] - 𝐐[𝐜[t], t])
        𝐐[3 - 𝐜[t], t+1] = 𝐐[3 - 𝐜[t], t] # retain value of unpicked choice
    end

    return 𝐜, 𝐫
end

# %%
"""
α: learning rate
β: inverse temperature
𝐜: vector of choices in each Nₜ trial in 1(A) or 2(B)
𝐫: 0 (no reward) or 1 (reward) in each Nₜ trial
"""
function func_qlearning(α, β, 𝐜, 𝐫)

    Nₜ = length(𝐜)
    Pₐ = zeros(Real, Nₜ) #probabilities of selecting A
    𝐐 = zeros(Real, (2, Nₜ))
    logl = 0 #initial value of log likelihood

    for t in 1:Nₜ - 1
        Pₐ[t] = softmax(β, 𝐐[1, t] - 𝐐[2, t])
        logl += (𝐜[t] == 1) * log(Pₐ[t]) + (𝐜[t] == 2) * log(1 - Pₐ[t])
        𝐐[𝐜[t], t + 1] = 𝐐[𝐜[t], t] + α * (𝐫[t] - 𝐐[𝐜[t], t])
        𝐐[3 - 𝐜[t], t + 1] =  𝐐[3 - 𝐜[t], t]
    end

    return (logl = logl, 𝐐 = 𝐐, Pₐ = Pₐ);
end

# %% [markdown]
# ## Parameter Estimation
# ### optimization with JuMP and Ipopt

# %%
using JuMP, Ipopt, ForwardDiff, Statistics

function estimate_parameter_qlearning(𝐜, 𝐫)   
    func_qlearning_JuMP(α, β) = func_qlearning(α, β, 𝐜, 𝐫).logl
    
    m = Model(Ipopt.Optimizer)
    register(m, :func_qlearning_JuMP, 2, func_qlearning_JuMP, autodiff=true)

    @variable(m, 0.0 <= α <= 1.0, start=rand(), base_name = "learning_rate")
    @variable(m, 0.0 <= β <= 5.0, start=rand(), base_name = "inverse_temperature")

    @NLobjective(m, Max, func_qlearning_JuMP(α, β))
    optimize!(m);
    #print(""," α = ", value(α), " β = ", value(β))

    print(""," α = ", α, " β = ", β)
    return value(α), value(β)
end

# %%
Nₜ= 500
α1 = 0.3
β1 = 0.5
Pᵣ = 0.5
𝐜, 𝐫 = generate_qlearning_data(Nₜ, α1, β1, Pᵣ)

# %%
estimate_parameter_qlearning(𝐜, 𝐫)

# %%

# %% [markdown]
# ## comparison of models
# ### win-stay lose-shift (WSLS) model

# %%
"""
Nₜ: number of trials
ϵ: error rate
Pᵣ: probability of getting reward in A
"""
function wsls_simulation(Nₜ, ϵ, Pᵣ, seed=1234)

    rng = MersenneTwister(seed)

    Pₐ = zeros(Nₜ) #probabilities of selecting A
    Pₐ[1] = 0.5 # probability at initial trial is 0.5
    𝐜 = zeros(Int, Nₜ) #initial choice in each Nₜ trial
    𝐫 = zeros(Nₜ) # 0 (no reward) or 1 (reward) in each Nₜ trial

    for t in 1:Nₜ-1
        chooseAB = rand(rng)
        get_reward = rand(rng)
        
        #select A with reward
        if chooseAB < Pₐ[t] && get_reward <  Pᵣ
            Pₐ[t + 1] = 1 - ϵ
            𝐜[t] = 1
            𝐫[t] = 1

        #select B with no reward
        elseif chooseAB > Pₐ[t] && get_reward >  Pᵣ
            Pₐ[t + 1] = 1 - ϵ
            𝐜[t] = 2
            𝐫[t] = 0

        #select A with no reward
        elseif chooseAB < Pₐ[t] && get_reward >  Pᵣ
            Pₐ[t + 1] = ϵ
            𝐜[t] = 1
            𝐫[t] = 0
        #select B with reward
        elseif chooseAB > Pₐ[t] && get_reward <  Pᵣ
            Pₐ[t + 1] = ϵ
            𝐜[t] = 2
            𝐫[t] = 1
        end
    end
    return (Pₐ = Pₐ, 𝐜 = 𝐜, 𝐫 = 𝐫);
end

# %%
@manipulate for Nₜ in 0:5:200, ϵ in 0:0.05:1, Pᵣ in 0:0.05:1, seed in 1:1:1234

    Pₐ = wsls_simulation(Nₜ, ϵ, Pᵣ, seed).Pₐ

    plot(Pₐ, label="P(a = A)", color="orange")
    ylabel!("P(a = A)")
    ylims!((0, 1))
    title!("WSLS Model")

end

# %% [markdown]
# ### random selection model

# %%
function random_choice_simulation(Nₜ, Pₐ, seed=1234)

    rng = MersenneTwister(seed)

    𝐜 = 2 .- Int.(rand(rng, Nₜ) .< Pₐ) #dot notation in Julia signifies elemnet-wise operation

    return (Pₐ = Pₐ, 𝐜 = 𝐜)
end

# %% [markdown]
# ### Interactive plot

# %%
@manipulate for Nₜ in 0:5:200, Pₐ in 0:0.05:1

    plot([Pₐ for i in range(1, stop=Nₜ)], label="P(a = A)", color="orange")
    ylabel!("P(a = A)")
    ylims!((0, 1))
    title!("Random Choice Model")

end

# %% [markdown]
# ### model comparison
# #### preparation

# %%
"""
ϵ: error rate
𝐜: vector of choices in each Nₜ trial in 1(A) or 2(B)
𝐫: 0 (no reward) or 1 (reward) in each Nₜ trial

when given ϵ, 𝐜, and 𝐫, returns log likelihood and Pₐ
"""
function func_wsls(ϵ, 𝐜, 𝐫)

    Nₜ = length(𝐜)
    Pₐ = zeros(Nₜ) #probabilities of selecting A
    Pₐ[1] = 0.5
    logl = 0.0 #initial value of log likelihood

    for t in 1:Nₜ - 1
        logl += (𝐜[t] == 1) * log(Pₐ[t]) + (𝐜[t] == 2) * log(1 - Pₐ[t])

        #select A with reward
        if 𝐜[t] == 1 &&   𝐫[t] == 1
            Pₐ[t + 1] = 1 - ϵ

        #select B with no reward
        elseif  𝐜[t] == 2 &&   𝐫[t] == 0
            Pₐ[t + 1] = 1 - ϵ

        #select A with no reward
        elseif  𝐜[t] == 1 &&   𝐫[t] == 0
            Pₐ[t + 1] = ϵ

        #select B with reward
        elseif 𝐜[t] == 2 &&   𝐫[t] == 1
            Pₐ[t + 1] = ϵ

        end
    end

    return (nlogl = -logl, Pₐ = Pₐ);
end

# %%
using Optim

function estimate_parameter_wsls(𝐜, 𝐫)   

    func_wsls_opt(ϵ) = func_wsls(ϵ, 𝐜, 𝐫).logl # why can't I use nlogl?

    result = optimize(func_wsls_opt, 0.0, 1.0)
    print(result)
    return Optim.minimizer(result)
end

# %%
estimate_parameter_wsls(𝐜, 𝐫)

# %%
wsls_data = wsls_simulation(200, 0.2, 0.5, 1234) 
𝐜, 𝐫  = wsls_data[1], wsls_data[2]
estimate_parameter_wsls(𝐜, 𝐫)

# %%
using Plots
plot([0:0.01:1], [func_wsls(i, 𝐜, 𝐫).logl for i in 0:0.01:1])
# %%
"""
Pₐ: probability of choosing A
𝐜: vector of choices in each Nₜ trial in 1(A) or 2(B)
𝐫: 0 (no reward) or 1 (reward) in each Nₜ trial

when given Pₐ, 𝐜, and 𝐫, returns log likelihood and Pₐ
"""
function func_random_choice(Pₐ, 𝐜, 𝐫)

    Nₜ = length(𝐜)
    logl = 0

    for t in 1:Nₜ
        logl += (𝐜[t] == 1) * log(Pₐ) + (𝐜[t] == 2) * log(1 - Pₐ)
    end

    return logl

end

# %% [markdown]
# #### compare log-likelihood and AIC of each model

# %%
Nₜ= 500
α1 = 0.3
β1 = 2
Pᵣ = 0.5

𝐜, 𝐫 = generate_qlearning_data(Nₜ, α1, β1, Pᵣ)

# Q-learning model
α, β = estimate_parameter_qlearning(𝐜, 𝐫)


# %%

# %%

# %%

# %%
