        fig, ax = plt.subplots()
        # Plot the lines with different colors and linestyles
        ax.plot(x, train_class_accs[:,0],linewidth=1, color='black', linestyle='--',label='base')
        # ax.plot(x, eval_class_accs[:,0], color='lightgreen', linestyle='dotted')
        ax.plot(x, train_class_accs[:,1],linewidth=1, color='green', linestyle='--',label='translation')
        # ax.plot(x, eval_class_accs[:,1], color='lightblue', linestyle='dotted')
        ax.plot(x, train_class_accs[:,2],linewidth=1, color='darkred', linestyle='--',label='rotation')
        # ax.plot(x, eval_class_accs[:,2], color='salmon', linestyle='dotted')
        ax.plot(x, train_accs           ,linewidth=2, color='goldenrod', linestyle='-',label='average')

        # # Plot an invisible line to create a label for the blue dashed line
        train_legend = mlines.Line2D([], [], color='black',linestyle='--',
                          markersize=15, label='training')
        test_legend = mlines.Line2D([], [], color='black', linestyle='dotted',
                          markersize=15, label='testing')
        mean_legend = mlines.Line2D([], [], color='black', linestyle='-',
                          markersize=15, label='mean')
        base_legend = mlines.Line2D([], [], color='gray',linestyle='-',
                          markersize=15, label='base')
        shift_legend=mlines.Line2D([], [], color='green',linestyle='-',
                          markersize=15, label='translation')
        rot_legend = mlines.Line2D([], [], color='red',linestyle='-',
                          markersize=15, label='rotation')
        ax.legend(handles=[train_legend,test_legend,mean_legend,base_legend,shift_legend,rot_legend])

        # Add legend and labels
        ax.legend()
        ax.set_xlabel('epochs')
        ax.set_ylabel('accuracy')
        ax.grid()

        # Display the plot
        plt.show()