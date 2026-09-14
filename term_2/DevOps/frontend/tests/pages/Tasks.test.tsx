import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import TasksPage from '../../pages/Tasks';
import { AppData, Task, Priority, Area, SearchResult, Note } from '../../types';
import { api } from '../../services/api';

vi.mock('../../services/api');

const mockAppData: AppData = {
  user: { id: 1, full_name: 'Test User', email: 'test@example.com' },
  areas: [
    { id: 1, name: 'Work', color: 'bg-red-500' },
    { id: 2, name: 'Personal', color: 'bg-blue-500' },
  ],
  tasks: [
    { id: 't1', title: 'Work Task High (Due Today)', area_id: 1, completed: false, description: 'High priority work task', due_date: new Date().toISOString().split('T')[0], priority: Priority.High, created_at: '2026-04-15T10:00:00Z' },
    { id: 't2', title: 'Personal Task Medium', area_id: 2, completed: false, description: 'Medium priority personal task', due_date: '2026-04-20', priority: Priority.Medium, created_at: '2026-04-14T10:00:00Z' },
    { id: 't3', title: 'Work Task Low (No Due Date)', area_id: 1, completed: false, description: 'Low priority work task', due_date: '', priority: Priority.Low, created_at: '2026-04-13T10:00:00Z' },
    { id: 't4', title: 'Completed Work Task', area_id: 1, completed: true, description: 'This task is done', due_date: '2026-04-10', priority: Priority.High, created_at: '2026-04-09T10:00:00Z' },
    { id: 't5', title: 'Personal Task High (Due Tomorrow)', area_id: 2, completed: false, description: 'High priority personal task', due_date: new Date(new Date().setDate(new Date().getDate() + 1)).toISOString().split('T')[0], priority: Priority.High, created_at: '2026-04-16T10:00:00Z' },
  ],
  notes: [], // Not directly used in TasksPage for display, but included in AppData
};

const waitForDebounce = () => new Promise(resolve => setTimeout(resolve, 350));

describe('Tasks Page', () => {
  const onDataUpdate = vi.fn();
  const onOpenTask = vi.fn();
  const onNewTask = vi.fn();
  const onOpenNote = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
    vi.useFakeTimers();
    (api.search as vi.Mock).mockResolvedValue([]);
    (api.toggleTaskCompletion as vi.Mock).mockResolvedValue({});
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  const renderComponent = (data: AppData | null = mockAppData) => {
    render(
      <TasksPage
        data={data}
        onDataUpdate={onDataUpdate}
        onOpenTask={onOpenTask}
        onNewTask={onNewTask}
        onOpenNote={onOpenNote}
      />
    );
  };

  it('renders loading message when data is null', () => {
    renderComponent(null);
    expect(screen.getByText('Loading...')).toBeInTheDocument();
  });

  it('renders tasks correctly with data, grouped by area', () => {
    renderComponent();
    expect(screen.getByText('My Tasks')).toBeInTheDocument();
    expect(screen.getByText('Work Area')).toBeInTheDocument();
    expect(screen.getByText('Personal Area')).toBeInTheDocument();
    expect(screen.getByText('Work Task High (Due Today)')).toBeInTheDocument();
    expect(screen.getByText('Personal Task Medium')).toBeInTheDocument();
    expect(screen.getByText('Completed Tasks (1)')).toBeInTheDocument();
  });

  it('calls onNewTask when "Add Task" button is clicked', async () => {
    renderComponent();
    await userEvent.click(screen.getByRole('button', { name: /Add Task/i }));
    expect(onNewTask).toHaveBeenCalledTimes(1);
    expect(onNewTask).toHaveBeenCalledWith(null);
  });

  it('calls onOpenTask when a pending task is clicked', async () => {
    renderComponent();
    await userEvent.click(screen.getByText('Work Task High (Due Today)'));
    expect(onOpenTask).toHaveBeenCalledTimes(1);
    expect(onOpenTask).toHaveBeenCalledWith(mockAppData.tasks[0]);
  });

  it('calls api.toggleTaskCompletion and onDataUpdate when pending task checkbox is clicked', async () => {
    renderComponent();
    const taskTitle = screen.getByText('Work Task High (Due Today)');
    // Find the parent div of the task title, then query for the checkbox within it
    const taskCard = taskTitle.closest('.group');
    if (taskCard) {
      const checkbox = taskCard.querySelector('.w-5.h-5'); // Assuming this selects the checkbox-like div
      if (checkbox) {
        await userEvent.click(checkbox);
        expect(api.toggleTaskCompletion).toHaveBeenCalledTimes(1);
        expect(api.toggleTaskCompletion).toHaveBeenCalledWith('t1', false); // Toggling from false to true
        expect(onDataUpdate).toHaveBeenCalledTimes(1);
      } else {
        throw new Error("Could not find checkbox for task 'Work Task High (Due Today)'");
      }
    } else {
      throw new Error("Could not find task card for 'Work Task High (Due Today)'");
    }
  });

  it('calls api.toggleTaskCompletion and onDataUpdate when completed task is clicked to uncomplete', async () => {
    renderComponent();
    await userEvent.click(screen.getByText('Completed Work Task')); // Click on the completed task card
    expect(api.toggleTaskCompletion).toHaveBeenCalledTimes(1);
    expect(api.toggleTaskCompletion).toHaveBeenCalledWith('t4', true); // Toggling from true to false
    expect(onDataUpdate).toHaveBeenCalledTimes(1);
  });

  it('sorts tasks correctly within areas (priority, then due date)', () => {
    renderComponent();
    // Work Area tasks: High (t1), Low (t3)
    // Personal Area tasks: High (t5), Medium (t2)
    const workArea = screen.getByText('Work Area').closest('section');
    const personalArea = screen.getByText('Personal Area').closest('section');

    expect(workArea).toHaveTextContent('Work Task High (Due Today)');
    expect(workArea).toHaveTextContent('Work Task Low (No Due Date)');
    expect(personalArea).toHaveTextContent('Personal Task High (Due Tomorrow)');
    expect(personalArea).toHaveTextContent('Personal Task Medium');
  });

  it('handles search query and displays results', async () => {
    const searchResult: SearchResult[] = [{ id: 't1', type: 'task', title: 'Work Task High (Due Today)', description: 'High priority work task' }];
    (api.search as vi.Mock).mockResolvedValue(searchResult);

    renderComponent();
    const searchInput = screen.getByPlaceholderText('Search tasks...');
    
    await userEvent.type(searchInput, 'Work High');
    vi.advanceTimersByTime(300);

    await waitFor(() => {
      expect(api.search).toHaveBeenCalledWith('Work High', 'task');
      expect(screen.getByText('Work Task High (Due Today)')).toBeInTheDocument();
    });

    await userEvent.click(screen.getByText('Work Task High (Due Today)'));
    expect(onOpenTask).toHaveBeenCalledTimes(1);
    expect(onOpenTask).toHaveBeenCalledWith(mockAppData.tasks[0]);
    expect(screen.queryByText('Work Task High (Due Today)')).not.toBeInTheDocument(); // Search results should disappear
  });

  it('displays "No results found." when search returns empty', async () => {
    (api.search as vi.Mock).mockResolvedValue([]);

    renderComponent();
    const searchInput = screen.getByPlaceholderText('Search tasks...');
    
    await userEvent.type(searchInput, 'NonExistent');
    vi.advanceTimersByTime(300);

    await waitFor(() => {
      expect(api.search).toHaveBeenCalledWith('NonExistent', 'task');
      expect(screen.getByText('No results found.')).toBeInTheDocument();
    });
  });

  it('clears search results and query on click outside', async () => {
    const searchResult: SearchResult[] = [{ id: 't1', type: 'task', title: 'Work Task High (Due Today)', description: 'High priority work task' }];
    (api.search as vi.Mock).mockResolvedValue(searchResult);

    renderComponent();
    const searchInput = screen.getByPlaceholderText('Search tasks...');
    
    await userEvent.type(searchInput, 'Work');
    vi.advanceTimersByTime(300);

    await waitFor(() => {
      expect(screen.getByText('Work Task High (Due Today)')).toBeInTheDocument();
    });

    await userEvent.click(document.body);
    expect(screen.queryByText('Work Task High (Due Today)')).not.toBeInTheDocument();
    expect(searchInput).toHaveValue('');
  });

  it('displays completed tasks in a separate section', () => {
    renderComponent();
    const completedTasksSection = screen.getByText('Completed Tasks (1)').closest('section');
    expect(completedTasksSection).toBeInTheDocument();
    expect(completedTasksSection).toHaveTextContent('Completed Work Task');
    expect(screen.queryByText('Work Task High (Due Today)')).not.toBeInTheDocument(); // Should not be in completed section
  });

  it('does not display empty area sections', () => {
    const appDataWithEmptyArea = {
      ...mockAppData,
      tasks: [mockAppData.tasks[3]], // Only completed task
      areas: [{ id: 3, name: 'Empty Area', color: 'bg-green-500' }, ...mockAppData.areas],
    };
    renderComponent(appDataWithEmptyArea);
    expect(screen.queryByText('Empty Area')).not.toBeInTheDocument();
    expect(screen.getByText('Completed Tasks (1)')).toBeInTheDocument();
  });
});