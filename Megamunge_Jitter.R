#' Megamunge
#'This funcion is meant to take the entire signal dataframe from the emg data from the MATLAB AE, tidy it, 
#'and plot it with ggplot
#' 
#' @param Grip_Type.csv 
#'
#' @return plot of signals
#' @export
#'
#' @examples

Megamunge <- function(fname){
  df <- read_csv(fname)

#Simplify Electrode ID's
colnames(df) <- gsub("rawDataOut", "E", colnames(df))

## Tidy Pipes
  df %>%
    select(E1:E8) %>%
    rownames_to_column(var = "Electrode") %>% 
    gather(Electrode, Readout) -> Grip_Type_Tidy

  #View(Grip_Type_Tidy)

## Generate plot titles based on file names
plot_title <- paste("Signal Intensities for", toString(fname))
plot_title <- gsub(".csv", "", plot_title)
                      
#### Data Visualization
ggplot(Grip_Type_Tidy, aes(x = Grip_Type_Tidy$Electrode, y=Readout, color = Readout))+ 
    geom_jitter() +xlab("Electrode (E)") + ylab("Readout (mV)") + ggtitle(plot_title)
  
}


