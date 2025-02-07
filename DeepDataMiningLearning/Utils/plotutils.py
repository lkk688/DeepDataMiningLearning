import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

figsize = (12, 8)
fontsize = 15
outputfolder = 'output/'

def plotcomparisonbar(data1, data2, labels, tick_labels, xlabel='Questions', ylabel='Time Taken (s)', figname='answeringcomparision.pdf'):
    # Set width of bar
    barWidth = 0.25
    fig = plt.subplots(figsize =figsize)

    # Set position of bar on X axis
    br1 = np.arange(len(data1))
    br2 = [x + barWidth for x in br1]

    # Make the plot
    plt.bar(br1, data1, color ='r', width = barWidth,
            edgecolor ='grey', label =labels[0])
    plt.bar(br2, data2, color ='g', width = barWidth,
            edgecolor ='grey', label =labels[1])

    plt.xticks([r + barWidth for r in range(len(data1))], tick_labels, rotation=0, ha='right', fontsize = fontsize)

    plt.xlabel(xlabel, fontweight ='bold', fontsize = fontsize)
    plt.ylabel(ylabel, fontweight ='bold', fontsize = fontsize)
    plt.legend()

    # Annotate bars with labels
    for i, (d1, d2) in enumerate(zip(data1, data2)):
        plt.text(br1[i], d1, f'{d1:.2f}', ha='center', va='bottom', color='black', fontweight='bold')
        plt.text(br2[i], d2, f'{d2:.2f}', ha='center', va='bottom', color='black', fontweight='bold')

    plt.tight_layout()  # Adjust layout to prevent clipping
    plt.savefig(os.path.join(outputfolder, figname), format='pdf')  # Save the figure as a pdf file
    plt.show()


def plotVectors(vecs, cols, alpha=1):
    """
    Plot set of vectors.

    Parameters
    ----------
    vecs : array-like
        Coordinates of the vectors to plot. Each vector is in an array. For
        instance: [[1, 3], [2, 2]] can be used to plot 2 vectors.
    cols : array-like
        Colors of the vectors. For instance: ['red', 'blue'] will display the
        first vector in red and the second in blue.
    alpha : float
        Opacity of vectors

    Returns:

    fig : instance of matplotlib.figure.Figure
        The figure of the vectors
    """
    plt.axvline(x=0, color='#A9A9A9', zorder=0)
    plt.axhline(y=0, color='#A9A9A9', zorder=0)

    for i in range(len(vecs)):
        if (isinstance(alpha, list)):
            alpha_i = alpha[i]
        else:
            alpha_i = alpha
        if (len(vecs[i])==2):
            x = np.concatenate([[0,0],vecs[i]])
        elif (len(vecs[i])==4):
            x = vecs[i]
        plt.quiver([x[0]],
                   [x[1]],
                   [x[2]],
                   [x[3]],
                   angles='xy', scale_units='xy', scale=1, color=cols[i],
                  alpha=alpha_i)


def testcompare():
    #from https://colab.research.google.com/drive/1h-a6QwyMaIz7cJnsoaagQM1KTSU8fNCa#scrollTo=nqKJ6pkPuw3f
    plotcomparisonbar(data1=[12.775,  15.78,  49.5], data2=[13.367, 15.76, 51.77], \
                    labels=['FastAPI in Pods','Direct Pytorch Inference'],\
                    tick_labels=['Case 1', 'Case 2', 'Case 3'], xlabel='Questions', ylabel='Time Taken (s)', \
                    figname='answeringcomparision.pdf')
    
    """video transcription between fastapi and old implementation.
    Fast APi was better only in the smaller video length
    time increased as the video size increased."""
    plotcomparisonbar(data1=[6.48,15.87,25.825], data2=[10.12, 15.37, 24.61], \
                    labels=['FastAPI in Pods','Direct Pytorch Inference'],\
                    tick_labels=["1 min", "3.38 mins", "5.39mins"], xlabel='Video Length', ylabel='Time Taken (s)', \
                    figname='videotranscriptioncomparision.pdf')
    
    """audio transcption
    the above videos were converted to.mp3
    in this case fastapi was always better but there is also one more insight even thought the same code is
    used for video and audio. The timetaken by the audio files is much lower than the video files
    """

    plotcomparisonbar(data1=[4.8,12.6,21.36], data2=[6.03,13.99,21.9], \
                    labels=['FastAPI in Pods','Direct Pytorch Inference'],\
                    tick_labels=["1 min", "3.38 mins", "5.39mins"], xlabel='Audio Length', ylabel='Time Taken (s)', \
                    figname='audiotranscptioncomparision.pdf')
    
    plotcomparisonbar(data1=[5.593, 10.607, 25.608], data2=[10.599, 15.631, 35.616], \
                  labels=['FastAPI in Pods','Direct Pytorch Inference'],\
                  tick_labels=["Case1", "Case2", "Case2"], xlabel='Sentence Length', ylabel='Time Taken (s)', \
                  figname='Translationfromenglishtospanishcomparision.pdf')
    
    plotcomparisonbar(data1=[5.584,11.245,20.589], data2=[5.602,10.704,25.654], \
                  labels=['FastAPI in Pods','Direct Pytorch Inference'],\
                  tick_labels=["Case1", "Case2", "Case2"], xlabel='Sentence Length', ylabel='Time Taken (s)', \
                  figname='Translationfromspanishtoenglishcomparision.pdf')

if __name__ == "__main__":
    errorModel1 = [22, -4]
    errorModel2 = [14, 9]
    plotVectors([errorModel1, errorModel2], [sns.color_palette()[0], sns.color_palette()[1]])

    plt.xlim(-1, 25)
    plt.ylim(-5, 10)
    plt.show()
    
    u = np.array([0,0,1,6])
    v = np.array([0,0,4,2])
    w = u+v

    u_bis = [u[2], u[3], v[2], v[3]]

    plotVectors([u, u_bis, w],
                [sns.color_palette()[0],
                sns.color_palette()[1],
                sns.color_palette()[2]])

    plt.xlim(-2, 6)
    plt.ylim(-2, 9)

    plt.text(-1, 3.5, r'$||\vec{u}||$', color=sns.color_palette()[0], size=20)
    plt.text(2.5, 7.5, r'$||\vec{v}||$', color=sns.color_palette()[1], size=20)
    plt.text(2, 2, r'$||\vec{u}+\vec{v}||$', color=sns.color_palette()[2], size=20)

    plt.show()
    plt.close()
    
    x = [0,0,0,2]
    y = [0,0,2,2]

    plotVectors([x, y], [sns.color_palette()[0], sns.color_palette()[1]])

    plt.xlim(-1, 3)
    plt.ylim(-1, 3)

    plt.text(-0.5, 1, r'$\vec{x}$', size=18, color=sns.color_palette()[0])
    plt.text(1.5, 0.5, r'$\vec{y}$', size=18, color=sns.color_palette()[1])
    
    plt.show()
    plt.close()


