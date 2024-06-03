import torch
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

######## endroit à compléter #########
# fonction de cout
def loss_msq(y,yh):
    #remplir ici
    return #

# fonction de regression
def forward(a,X):
    #remplir ici
    return #
    
######################################    

# creation donnees : modele + bruit
np.random.seed(0)
true_a = 2.0  # pente à trouver
n_points = 50
x = np.linspace(-1, 1, n_points)
bruit = np.random.normal(0, 0.1, n_points)
y = true_a * x + bruit

X = torch.tensor(x)
Y = torch.tensor(y)

# initialisation des parametres d'apprentissage
a = torch.tensor(np.random.randn(), requires_grad=True) 

# taille du pas d'apprentissage (learning rate)
lr = 0.07

# nombres de pas d'apprentissage    
epochs = 200

# methode d'optimisation 
optimizer = torch.optim.SGD((a,), lr =lr) 

# erreur et pente
Loss = np.zeros(epochs)
A    = np.zeros(epochs)


for i in range(epochs):
    
 
    # Forward pass: Calcul y predit en envoyant X dans le model
    Yh = forward(a,X)
 
    # Calcul loss (fonction de cout)
    loss = loss_msq(Y,Yh)
    
 
    Loss[i] = loss.detach().numpy()
    
    #update de a
    optimizer.zero_grad() # gradient reinitialisé
    loss.backward() # retropropagation du graddient : grad(loss_msq)
    optimizer.step() # a = a - lr* grad(loss_msq)
    
    ad = a.detach().numpy()
    A[i] =ad
    


########### Visualization des results ######################
fig, ax = plt.subplots(2, 1, figsize=(8, 12))

# Plotting error over iterations
ax[0].plot(Loss, label='MC')
ax[0].set_title('Erreur pendant les Iterations')
ax[0].set_xlabel('Iteration')
ax[0].set_ylabel('Moindre carré')
ax[0].legend()

# Plotting the evolution of a
ax[1].plot(A, label='Valeur de a')
ax[1].set_title('Evolution de la pente pendant les Iterations')
ax[1].set_xlabel('Iteration')
ax[1].set_ylabel('Valeur de a')
ax[1].legend()

# Show plots
plt.show()



# Setup the figure and axis
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(x, y, color='blue', label='Points')
ax.plot(x, true_a * x, 'g-', label=f'Vraie Droite : y = {true_a}x')

line, = ax.plot(x, A[0] * x, 'r-', linewidth=2, label=f'Droite apprise')
ax.legend()
ax.set_title('Convergence vers la meilleure correspondance')
ax.set_xlabel('x')
ax.set_ylabel('y')

# Animation update function
def update(frame):
    line.set_ydata(A[frame] * x)
    
    #line.set_label(f'Droite apprise: y = {A[frame]:.2f}x')
    ax.legend().remove()  # Remove the existing legend
    ax.legend(loc='upper left')  # Update the legend to show the new label
    
    return line,

# Create animation
ani = FuncAnimation(fig, update, frames=epochs, blit=True, repeat=False)

# Pour sauver l'animation, decommenter cette ligne:
# ani.save('droite_convergence.mp4', writer='ffmpeg')

plt.show()





