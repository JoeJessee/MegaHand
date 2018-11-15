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
## Tidy Pipes
  df %>%
    select(rawDataOut1:rawDataOut8) %>% 
    rownames_to_column(var = "Electrode") %>% 
    gather(Electrode, Readout) -> Grip_Type_Tidy

  #View(Grip_Type_Tidy)
  
  
#### Data Visualization
ggplot(Grip_Type_Tidy, aes(x = Grip_Type_Tidy$Electrode, y=Readout, color = Readout))+ 
    geom_jitter() +xlab("Electrode") + ylab("Some unit") + ggtitle("Signal Intensity")
  
}


